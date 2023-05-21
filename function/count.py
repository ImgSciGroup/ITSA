def count_white_black(image1,image2,image3):
    t23_white = 0
    t12_white = 0
    t12_black = 0
    t23_black = 0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            (b, g, r) = image1[i, j]
            (b1, g1, r1) = image2[i, j]
            if ((b, g, r) == (b1, g1, r1)==(0,0,0)):
                t12_black+= 1
            if ((b, g, r) == (b1, g1, r1)==(255,255,255)):
                t12_white+= 1
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            (b, g, r) = image3[i, j]
            (b1, g1, r1) = image2[i, j]
            if ((b, g, r) == (b1, g1, r1)==(0,0,0)):
                t23_black += 1
            if ((b, g, r) == (b1, g1, r1)==(255,255,255)):
                t23_white += 1
    return (abs(t12_black-t23_black)/(image1.shape[0]*image1.shape[1])),(abs(t12_white-t23_white)/(image1.shape[0]*image1.shape[1]))