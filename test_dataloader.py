from datahandling import dataloader as dl


loader = dl.DataLoader(directory="rep22", type_stat="statistiques")
loader.load_data()


print(loader[0,0,0].shape)
print(loader[0,0].shape)
print(loader[0].shape)

print(loader["T"].shape)

loader = dl.DataLoader(columns=[0,"T"])
loader.load_data()
print(loader.shape)