#!/bin/bash
jupyter lab stop
nohup poetry run jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.token=groups --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* &> jupyter.log &
