import numpy as np


def score(gold_matrices, pred_matrices, do_print=False, debug=False):
    tp, fp, tn, fn = 0, 0, 0, 0
    tp_ = 0
    em, em_ = 0, 0
    tot = 0
    for gmgl, pmpl in zip(gold_matrices, pred_matrices):
        gl = gmgl
        pl = pmpl
        gm = np.where(gl < 1, gl, 1) 
        pm = np.where(pl < 1, pl, 1)
        tot += 1
        #print(gmgl.shape)
        #print(pmpl.shape)
        n = len(gm)
        assert gm.shape == pm.shape, "different matrix shapes"
        if np.all(np.equal(gm, pm)):
            em += 1
        if np.all(np.equal(gl, pl)):
            em_ += 1
        for i in range(n):
            for j in range(n):
                if gm[i,j] and pm[i,j]:
                    tp += 1
                    if gl[i,j] == pl[i,j]:
                        tp_ += 1
                elif gm[i,j] and not pm[i,j]:
                    fn += 1
                elif not gm[i,j] and pm[i,j]:
                    fp += 1
                elif not gm[i,j] and not pm[i,j]:
                    tn += 1

    if do_print:
        print(tp, fp, fn)
        print(tp_, fp, fn)
    results = {}
    p, r, f = 0, 0, 0
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        pass
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        pass
    try:
        f = 2 * p * r / (p + r)
    except ZeroDivisionError:
        pass
    if do_print:
        print("UP: {:.2%}\tUR: {:.2%}\tUF: {:.2%}".format(p, r, f))
        print("UEM: {:.2%}".format(em / tot))
    results["UP"] = p
    results["UR"] = r
    results["UF"] = f
    results["UEM"] = em / tot
    
    lf = f

    p, r, f = 0, 0, 0
    try:
        p = tp_ / (tp + fp)
    except ZeroDivisionError:
        pass
    try:
        r = tp_ / (tp + fn)
    except ZeroDivisionError:
        pass
    try:
        f = 2 * p * r / (p + r)
    except ZeroDivisionError:
        pass
    if do_print:
        print("LP: {:.2%}\tLR: {:.2%}\tLF: {:.2%}".format(p, r, f))
    try:
        if do_print:
            print("LEM: {:.2%}".format(em_ / tot))
        results["LEM"] = em_ / tot
    except ZeroDivisionError:
        if do_print:
            print("LEM: {:.2%}".format(0))
        results["LEM"] = 0
    try:
        if do_print:
            print("LA: {:.2%}".format(tp_ / tp))
        results["LA"] = tp_ / tp
    except ZeroDivisionError:
        if do_print:
            print("LA: {:.2%}".format(0))
        results["LA"] = 0
    
    results["LP"] = p
    results["LR"] = r
    results["LF"] = f
    lf = f
    
    if debug:
        print(f"{p:.2f}, {r:.2f}, {f:.2f}")
        print(tp, tp_, fp, fn)
    return lf, results
