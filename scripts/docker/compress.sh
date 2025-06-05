nvidia-docker run --rm -it -v "$(pwd)":"/home/"  \
    -w "/home/"  \
    --entrypoint nvidia-smi \
    nif:compress
    # compress.py configurations/.compression/vanilla.toml
