import argparse
import copy
import ipaddress
import itertools
import json
import logging
import os
import threading
from argparse import Namespace
from typing import List, Optional

import aiohttp
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from vllm.entrypoints.utils import with_cancellation

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

prefill_instances: Optional[
    List[str]] = None  # List of url of the prefill vllm instances
decode_instances: Optional[
    List[str]] = None  # List of url of the decode vllm instances
tokenizer = None

prefill_cycler = None
decode_cycler = None
lock = threading.Lock()

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def safe_next(cycler: itertools.cycle):
    with lock:
        return next(cycler)


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: Only 'application/json' is allowed"
        )


router = APIRouter()


async def forward_request(url, data, use_chunked=True):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        try:
            async with session.post(url=url, json=data,
                                    headers=headers) as response:
                if 200 <= response.status < 300 or 400 <= response.status < 500:
                    if use_chunked:
                        async for chunk_bytes in response.content.iter_chunked(
                                1024):
                            yield chunk_bytes
                    else:
                        content = await response.read()
                        yield content
                else:
                    error_content = await response.text()
                    try:
                        error_content = json.loads(error_content)
                    except json.JSONDecodeError:
                        error_content = error_content
                    logger.error(
                        f"Request failed with status {response.status}: {error_content}"
                    )
                    raise HTTPException(
                        status_code=response.status,
                        detail=
                        f"Request failed with status {response.status}: {error_content}"
                    )
        except aiohttp.ClientError as e:
            logger.error(f"ClientError occurred: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail="Bad Gateway: Error communicating with upstream server."
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


def calculate_id_and_prefix_hash(prompt):
    tokens = tokenizer.tokenize(prompt)
    prompt_ids = tokenizer.convert_tokens_to_ids(tokens)
    keys = []
    prefix_key = str(hash(prompt))
    keys.append(prefix_key)
    return prompt_ids, keys


def get_ids_and_tokens_from_prompts(prompt):
    if prompt is List:
        prompts_ids = []
        all_keys = []
        for p in prompt:
            prompt_ids, keys = calculate_id_and_prefix_hash(p)
            prompts_ids.append(prompt_ids)
            all_keys.append(keys)
        return prompts_ids, all_keys
    else:
        return calculate_id_and_prefix_hash(prompt)


@with_cancellation
@router.post('/v1/completions', dependencies=[Depends(validate_json_request)])
async def create_completion(raw_request: Request):
    try:
        request = await raw_request.json()
        # get ids and keys
        prefix_prompt_ids, keys = get_ids_and_tokens_from_prompts(
            request["prompt"])
        # add params to request
        kv_prepare_request = request.copy()
        kv_prepare_request["max_tokens"] = 1
        kv_transfer_params = {
            "prefix_prompt_ids": prefix_prompt_ids,
            "kvcache_store_keys": keys,
            "kvcache_load_keys": None,
        }
        kv_prepare_request["kv_transfer_params"] = kv_transfer_params
        # prefill stage
        prefill_instance = safe_next(prefill_cycler)
        async for _ in forward_request('http://'\
            + prefill_instance + '/v1/completions', kv_prepare_request):
            continue

        # Perform kv recv and decoding stage
        decode_instance = safe_next(decode_cycler)
        kv_transfer_params = {
            "prefix_prompt_ids": prefix_prompt_ids,
            "kvcache_store_keys": None,
            "kvcache_load_keys": keys,
        }
        request["kv_transfer_params"] = kv_transfer_params
        generator = forward_request('http://'\
            + decode_instance + '/v1/completions', request)
        response = StreamingResponse(generator)
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception:
        import sys
        exc_info = sys.exc_info()
        print('Error occurred in disagg proxy server')
        print(exc_info)


@router.post('/v1/chat/completions',
             dependencies=[Depends(validate_json_request)])
async def create_chat_completion(raw_request: Request):
    try:
        request = await raw_request.json()
        # get ids and keys
        prompt = tokenizer.apply_chat_template(request["messages"],
                                               tokenize=False)
        prefix_prompt_ids, keys = get_ids_and_tokens_from_prompts(prompt)
        # add params to request
        kv_prepare_request = request.copy()
        kv_prepare_request["max_tokens"] = 1
        kv_transfer_params = {
            "prefix_prompt_ids": prefix_prompt_ids,
            "kvcache_store_keys": keys,
            "kvcache_load_keys": None,
        }
        kv_prepare_request["kv_transfer_params"] = kv_transfer_params
        # prefill stage
        prefill_instance = safe_next(prefill_cycler)
        async for _ in forward_request('http://'\
            + prefill_instance + '/v1/chat/completions', kv_prepare_request):
            continue
        # Perform kv recv and decoding stage
        decode_instance = safe_next(decode_cycler)
        kv_transfer_params = {
            "prefix_prompt_ids": prefix_prompt_ids,
            "kvcache_store_keys": None,
            "kvcache_load_keys": keys,
        }
        request["kv_transfer_params"] = kv_transfer_params
        generator = forward_request('http://'\
            + decode_instance + '/v1/chat/completions', request)
        response = StreamingResponse(content=generator)
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception:
        import sys
        exc_info = sys.exc_info()
        print('Error occurred in disagg proxy server')
        print(exc_info)
        return StreamingResponse(content=exc_info,
                                 media_type="text/event-stream")


def run_server(args, **uvicorn_kwargs) -> None:
    logger.info('vLLM disaggregated proxy start.')
    logger.info('args: %s', args)

    global prefill_instances, decode_instances, general_instances
    prefill_instances = args.prefill
    decode_instances = args.decode

    app = build_app(args)

    config = uvicorn.Config(app, port=args.port, loop='uvloop')
    server = uvicorn.Server(config)
    server.run()


def build_app(args: Namespace) -> FastAPI:

    app = FastAPI()
    app.include_router(router)

    return app


def verify_model_config(instances: List[str], model: str) -> None:
    for instance in instances:
        response = requests.get('http://' + instance + '/v1/models')
        if response.status_code == 200:
            model_cur = response.json()['data'][0]['id']
            if model_cur != model:
                raise ValueError(f'{instance} serve with the different model!')
        else:
            raise ValueError(f'Can not get model id from {instance}!')


def validate_instances(instances: List[str]):
    for instance in instances:
        if len(instance.split(':')) == 2:
            host, port = instance.split(':')
            try:
                if host != 'localhost':
                    ipaddress.ip_address(host)
                port = int(port)
                if port < 0 or port > 65535:
                    raise ValueError(f'Invalid instance: {instance}')
            except ValueError as err:
                raise ValueError(f'Invalid instance: {instance}') from err
        else:
            raise ValueError(f'Invalid instance: {instance}')


def validate_parsed_serve_args(args: Namespace):
    if args.prefill is None:
        raise ValueError('Please at least specify a prefill node.')
    else:
        validate_instances(args.prefill)
        verify_model_config(args.prefill, args.model)
    if args.decode is None:
        raise ValueError('Please at least specify a decode node.')
    else:
        validate_instances(args.decode)
        verify_model_config(args.decode, args.model)
    global prefill_cycler, decode_cycler, tokenizer
    prefill_cycler = itertools.cycle(args.prefill)
    decode_cycler = itertools.cycle(args.decode)
    tokenizer = AutoTokenizer.from_pretrained(args.model)


if __name__ == '__main__':
    # Todo: allow more config
    parser = argparse.ArgumentParser('vLLM disaggregated proxy server.')
    parser.add_argument('--model',
                        '-m',
                        type=str,
                        required=True,
                        help='Model name')

    parser.add_argument('--prefill',
                        '-p',
                        type=str,
                        nargs='+',
                        help='List of prefill node URLs (host:port)')

    parser.add_argument('--decode',
                        '-d',
                        type=str,
                        nargs='+',
                        help='List of decode node URLs (host:port)')

    parser.add_argument('--port',
                        type=int,
                        default=8000,
                        help='Server port number')
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    run_server(args)
