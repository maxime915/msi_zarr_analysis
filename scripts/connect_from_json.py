import json

from cytomine import Cytomine


def connect(json_path: str = "config_cytomine.json") -> Cytomine:

    with open(json_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    return connect_with_params(host_url, pub_key, priv_key)


def connect_with_params(
    host_url: str,
    pub_key: str,
    priv_key: str,
) -> Cytomine:
    return Cytomine.connect(host_url, pub_key, priv_key)
