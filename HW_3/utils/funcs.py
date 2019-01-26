def find_middle(list: list) -> object:
    l = len(list)
    if l % 2 != 0:
        return list[l // 2 + 1]
    else:
        return (list[l - 1] + list[l]) / 2
