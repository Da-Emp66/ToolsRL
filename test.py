mylist = [-3, 5, 6, 83, 52]


for idx, _ in enumerate(mylist):
    print([mylist[idx]] + [mylist[(idx + 1) % len(mylist)]])