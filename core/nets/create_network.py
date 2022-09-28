import imp

from configs import cfg

def _query_network():
    module = cfg.network_module
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network
    return network


def create_network(canonical_joints=None):
    network = _query_network()
    network = network(canonical_joints=canonical_joints)
    return network
