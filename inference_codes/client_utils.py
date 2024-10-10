import logging
import yaml

logger = logging.getLogger(__name__)

SHARED_CODE_DIR = "/root/poc/inference_codes/.ambre"
SERVER_CONFIG_FILE = "server_config.yaml"
CURRENT_MODEL_CONFIG_FILE="current_model_config.yaml"

def get_server_config():

    fname = SHARED_CODE_DIR + "/" + SERVER_CONFIG_FILE
    try:
        with open(fname, "r") as f:
            try:
                server_config = yaml.safe_load(f)
                SERVER_IP = server_config["IP"]
                AGENT_PORT = server_config["AGENT_PORT"]
                SERVER_PORT = server_config["SERVER_PORT"]
                return f"http://{SERVER_IP}:{SERVER_PORT}"

            except yaml.YAMLError as e:
                logger.error(fname + " load error")
                return False
    except PermissionError as e:
        logger.error(fname + " not found")
        return False  
        
    except FileNotFoundError as e:
        logger.error(fname + " not found")
        return False  

def get_model_config():

    fname = SHARED_CODE_DIR + "/" + CURRENT_MODEL_CONFIG_FILE
    try:
        with open(fname, "r") as f:
            try:
                model_config = yaml.safe_load(f)
                model_name = model_config["name"]
                model_path = model_config["path"]
                return model_path
            
            except yaml.YAMLError as e:
                logger.error(fname + " load error")
                return False
            
    except PermissionError as e:
        logger.error(fname + " not found")
        return False  
        
    except FileNotFoundError as e:
        logger.error(fname + " not found")
        return False  