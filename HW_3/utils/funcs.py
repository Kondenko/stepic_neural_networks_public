def find_middle(list: list) -> object:
    l = len(list)
    if l % 2 != 0:
        return list[l // 2]
    else:
        return (list[l - 1] + list[l]) / 2


def save_to_file(agent, number=0):
    try:
        filename = "network_config_agent_%d_layers_%s.txt" % (
            number, "_".join(map(str, agent.neural_net.sizes)))
        agent.to_file(filename)
        print("Saved agent parameters to '%s'" % filename)
    except AttributeError:
        print("Error saving {} to file".format(agent))

