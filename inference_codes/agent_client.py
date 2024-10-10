import httpx
import sys
import yaml
import math
import time
import os
import argparse
from loguru import logger

SHARED_CKPT_DIR = "/root/poc/checkpoints"
SHARED_CODE_DIR = "/root/poc/inference_codes/.ambre"
SERVER_CONFIG_FILE = "server_config.yaml"

SERVER_IP: str
AGENT_PORT: int
SERVER_PORT: int


def get_server_config():
    global SERVER_IP
    global AGENT_PORT
    global SERVER_PORT

    fname = SHARED_CODE_DIR + "/" + SERVER_CONFIG_FILE
    with open(fname, "r") as f:
        try:
            server_config = yaml.safe_load(f)
            SERVER_IP = server_config["IP"]
            AGENT_PORT = server_config["AGENT_PORT"]
            SERVER_PORT = server_config["SERVER_PORT"]

        except yaml.YAMLError as e:
            logger.error(fname + " load error")


def check_server(client: httpx.Client):
    try:
        logger.info("Checking agent server status ...")
        _r = client.get("/models")
        check_response(_r)
        logger.info("Agent server is normal")
        print("")

    except httpx.RequestError as e:
        logger.error("An error occurred on agent server while requesting " +
                     str(e.request.url))
        logger.error("Please contact technical support immediately ")
        sys.exit()


def check_response(r: httpx.Response):
    if not r.is_success:
        # if r.is_success:
        logger.error("status code : " + str(r.status_code))
        logger.error("failed url : " + str(r.url))
        logger.error("failed request : " + str(r.request))
        logger.error("failed response text : " + str(r.text))
        sys.exit()
    return


def get_supported_models(client: httpx.Client):
    _r = client.get("/models")
    check_response(_r)
    return _r.json()


def get_current_model(client: httpx.Client):
    _r = client.get("/current_model")
    check_response(_r)
    return _r.json()


def print_current_model(res: dict):

    title = " Current Server Info. "
    model = " Model : " + res["name"] + " "
    lora = " LoRA : " + str(res["use_lora"]) + " "
    ckpt = " Checkpoint : " + res["ckpt_path"] + " "
    status = " Server Status : " + res["status"] + " "

    width = max(len(title), len(model), len(lora), len(ckpt), len(status)) + 2

    left = math.floor((width - len(title)) / 2)
    right = math.ceil((width - len(title)) / 2)
    print("┌" + "─" * left + title + "─" * right + "┐")
    print("│" + model + " " * (width - len(model)) + "│")
    print("│" + lora + " " * (width - len(lora)) + "│")
    print("│" + ckpt + " " * (width - len(ckpt)) + "│")
    print("│" + status + " " * (width - len(status)) + "│")
    print("└" + "─" * (width) + "┘")

    print("")
    print("")
    return


def set_current_model(req: dict, client: httpx.Client):
    r = client.put("/current_model", json=req)
    check_response(r)


def select_model(model_list: list) -> list:
    num_model = len(model_list)
    print("========== Supported Model List ==========")
    for idx, m in enumerate(model_list):
        print(" " + str(idx + 1) + ". " + m)
    print("==========================================")
    print("")

    while True:
        ret = input("Select Model Number [1-" + str(num_model) + "/q/Q] : ")
        if ret.isdigit() and (int(ret) > 0 and int(ret) <= num_model):
            time.sleep(0.4)
            print("Selected Model : " + model_list[int(ret) - 1])
            time.sleep(0.4)
            print("")
            print("")
            return model_list[int(ret) - 1]
        elif len(ret) == 1 and (ret == 'q' or ret == 'Q'):
            sys.exit()
        else:
            continue


def select_ckpt() -> tuple:
    print("========== Select Checkpoints ============")
    print(" 1. Use Pretrained Model (default model)")
    print(" 2. Use Your Checkpoint")
    print("==========================================")
    print("")
    while True:
        opt = input("Select Option [1-2/q/Q] : ")
        if opt.isdigit():
            if int(opt) == 1:
                print("")
                return True, ""
            elif int(opt) == 2:
                print("")
                ckpt_path = input("Chekcpoint path : " + SHARED_CKPT_DIR + "/")
                print("")
                return False, SHARED_CKPT_DIR + "/" + ckpt_path
            else:
                continue
        elif len(opt) == 1 and (opt == 'q' or opt == 'Q'):
            sys.exit()
        else:
            continue


def select_peft(ckpt_path: str) -> bool:
    print("============= Select PEFT ================")
    print(" Checkpoint path : " + ckpt_path)
    print("==========================================")
    print("")
    while True:
        opt = input("Dose this checkpoint use LoRA? [y/n/q/Q] : ")
        if len(opt) == 1:
            if opt == 'y':
                return True
            elif opt == 'n':
                return False
            elif opt == 'q' or opt == 'Q':
                sys.exit()


# TODO - add option for just check server status

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--check",
                        help="Check inference server staatus",
                        action="store_true")
    args = parser.parse_args()

    get_server_config()

    base_url = "http://" + SERVER_IP + ":" + str(AGENT_PORT)

    if args.check:
        with httpx.Client(base_url=base_url) as client:
            check_server(client)
            r = get_current_model(client)
            print_current_model(r)
        exit(0)

    with httpx.Client(base_url=base_url) as client:

        check_server(client)
        r = get_current_model(client)
        print_current_model(r)

        r = get_supported_models(client)
        model_name = select_model(list(r))

        use_pretrained, ckpt_path = select_ckpt()

        use_lora = False
        if not use_pretrained:
            use_lora = select_peft(ckpt_path)

        req = {
            "name": model_name,
            # "name" : "Test",
            # "name" : "Qwen2-72B-Instruct",
            "use_pretrained": use_pretrained,
            "ckpt_path": ckpt_path,
            # "ckpt_path" : "/app/model/Qwen2-72B-Instruct",
            # "ckpt_path" : "/app/model/Meta-Llama-3-8B-Instruct",
            # "ckpt_path": "/app/model/llama3-8b-it-lora-finetuned",
            "use_lora": use_lora
        }

        set_current_model(req, client)

        logger.info(" Request has been sent.")

        time.sleep(2)

        r = get_current_model(client)

        print(" Loading ", end="", flush=True)

        while r["status"] != "NORMAL":
            time.sleep(1)
            print(".", end="", flush=True)
            r = get_current_model(client)

        print("")

        logger.info(" Inference server has been successfully LOADED")
