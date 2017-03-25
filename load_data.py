def preprocess(num_levels=3, start_level=0, number_of_imgs = 1000):
    load_file = 'data/train_data/train_image.txt'
    label_file = 'data/train_data/train_label.txt'
    load_f = open(load_file, 'rb')
    label_f = open(label_file, 'rb')

    paths = load_f.readlines()
    labels = label_f.readlines()
    #print(paths[0])
    paths = [os.path.join(str('data/train_data/train/'),str(p.strip())[2:-1] ) for p in paths]
    #labels = [int(l) for l in labels]
    lbls = []
    #print paths


    imgs = []
    test_count = 0
    discarded = 0
    for i, path in enumerate(paths):
        if '5203' in path or '5930' in path:
            continue
        try:
            
            img = io.imread(path)
            img = rgb2hsv(img)
            img = img[:,:,2]
            lbls.append(int(labels[i])) 
        except OSError as err:
            continue

        # img = rgb2gray(img)

        # remove images which have 45% or more of 0.51+-0.1 (green when viz hsv)
        #green = 0.51
        #mask = (img < green + 0.1) & (img > green - 0.1)
        #ratio = np.count_nonzero(mask)/(img.shape[0]*img.shape[1])
        #if ratio >= 0.35:
            # discard images that have to much hsv-green
        #      discarded += 1
        #       continue




        preprocessed = preprocess_image(img, start_level, start_level + num_levels)
        imgs.append(preprocessed)
        if test_count % 200 == 0:
            print('Loaded '+str(test_count)+' images')
            gc.collect()
        test_count += 1
        if test_count >= number_of_imgs:
            break

    print('Positives: ', np.count_nonzero(lbls)/len(lbls))
    return imgs, lbls