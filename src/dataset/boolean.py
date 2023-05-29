"""Boolean dataset.

Generates boolean sequences and their solutions using Python's 
'eval()' method.

Typical usage example:
    
    dataset = BooleanDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for x, y in dataloader:
        print(f"{x = }")
        print(f"{y = }")
"""

# TODO