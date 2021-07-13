import numpy as np

def fscore(index, matrix):
    tp = matrix[index, index]
    tpfnfl = sum(matrix[index]) 
    fn = matrix[index,0]
    fl = tpfnfl - tp - fn
    fp = matrix[0, index]
    p = tp / (tpfnfl - fn + fp)
    r = tp / (tpfnfl)
    f = 2*p*r / (p+r)
    print(f"\ttp: {tp} fp: {fp} fn: {fn} fl: {fl}")
    print(f"\tp: {p:.2%}, r: {r:.2%}, f: {f:.2%}")
    return tp, fp, fn, fl, p, r, f

def confuse(gold_matrices, pred_matrices, i2w):

    C = np.zeros((len(i2w), len(i2w)))

    for gl, pl in zip(gold_matrices, pred_matrices):

        n = len(gl)
        assert gl.shape == pl.shape, "different matrix shapes"

        for i in range(n):
            for j in range(n):
                C[int(gl[i,j]), int(pl[i,j])] += 1
    # print(C)

    for i in range(4, len(C)):
        print(i2w[i])
        fscore(i,C)
        #for j in range(len(C)):
        #    print("\t", i2w[j], C[i,j])
    return C

if __name__ == "__main__":
    import col_data as cd
    import vocab as vcb
    import sys
    try:
        with open("vocabs.pk", "rb") as fh:
            vocabs = pickle.load(fh)
    except FileNotFoundError: 
        train = cd.read_col_data(sys.argv[1])
        _vocabs = vcb.make_vocabs(train, 0)
        vocabs = vcb.Vocabs(*_vocabs)

    gold = cd.read_col_data(sys.argv[2])
    pred = cd.read_col_data(sys.argv[3])
    gms = [g.make_matrix("scope", True, vocabs.scoperels.w2i) for g in gold]
    pms = [p.make_matrix("scope", True, vocabs.scoperels.w2i) for p in pred]
    confuse(gms, pms, vocabs.scoperels.i2w) 
