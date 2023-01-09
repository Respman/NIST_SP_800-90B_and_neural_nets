import json

def Parse_config(json_file):
    f = open(json_file, 'r')
    ctx = json.load(f)
    f.close()
    return ctx