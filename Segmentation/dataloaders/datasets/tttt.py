from jcai_region import JcaiSegmentation

train_set = JcaiSegmentation(args, split='train')
val_set = JcaiSegmentation(args, split='val')
num_class = train_set.NUM_CLASSES
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = None