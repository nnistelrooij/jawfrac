from typing import List

import matplotlib.pyplot as plt
import numpy as np


def compute_froc_curve(
    probs: List[List[float]],
    positives: List[List[int]],
    num_scans: int,
):
    assert len(probs) == len(positives)
    for prs, pos in zip(probs, positives):
        assert len(prs) == len(pos)

    all_positives = np.array([p for ps in positives for p in ps])
    num_positives = sum([p[0] == 1 for p in positives])

    all_probs = np.array([p for ps in probs for p in ps])
    all_positives = all_positives[all_probs >= 0]
    all_probs = all_probs[all_probs >= 0]

    recalls, fps = [], []
    threshs = np.concatenate((np.sort(all_probs), [1.0]))
    for thresh in threshs:
        recall = 0
        for ps, classes in zip(probs, positives):
            if classes[0] != 1:
                continue

            recall += np.any(ps >= thresh)

        mask = all_probs >= thresh
        fp = (mask.sum() - all_positives[mask].sum()) / num_scans

        recalls.append(recall / num_positives)
        fps.append(fp)

    fps.insert(0, 1)
    recalls.insert(0, recalls[0])

    fps = np.array(fps)[::-1]
    recalls = np.array(recalls)[::-1]

    sensitivies = []
    for level in [1/16, 1/8, 1/4, 1/2, 1]:
        idx = (fps >= level).argmax()
        sensitivies.append(recalls[idx])

    return fps, recalls, np.mean(sensitivies)

def load_nnunet_results():
    probs = [
        [0.80503577], [0.9534944], [0.9009149],
        [0.9207468], [0.9603174], [0.88321555],
        [0.9131631], [0.95335484],
        [0.9407993], [0.9700135],
        [0.9885335, 0.89323914],
        [0.96991366],
        [0.9876679, 0.9400838],
        [0.9901125], [0.8473411, 0.7780497, 0.5223485],
        [0.9540335],
        [0.96235967], [0.96271086],
        [0.9567827, 0.8982943],
        [0.95987344],
        [0.9660361], [0.94909215], [0.8916204],    
        [0.78310686, 0.8409856], [0.78700185, 0.93852663], [0.96300024, 0.9180765],
        [0.9714332], [0.62134993, 0.65393984],    
        [0.9639711],
        [0.96566427], [0.7450536], [0.984824], [0.8875911],
        [0.7277644], [0.9857333], [0.90797746], [0.94997704],
        [0.9703383],
        [0.9079107], [0.9669119],
        [0.876243], [0.9787087],
        [0.94686383, 0.96302456],    
        [0.9577993],
        [0.9404632, 0.9857793], [0.70765924, 0.5606133],
        [0.9557924],
        [0.96651745], [0.89841217, 0.8842499, 0.93986905],
        [0.9686492], [0.8260571, 0.61165226],
        [0.96408725], [0.9564014], [0.9645767],
        [0.8633857, 0.96387255], [0.9813358],    
        [0.9670006],
        [0.8376222, 0.5697765], [0.9689801], [0.6995978, 0.8807851, 0.91766614],
        [0.9679379],
        [0.94963235],
        [0.9832879],
        [0.93943584], [-1],
    ]

    positives = [
        [1], [1], [0],
        [1], [1], [0],
        [1], [1],
        [0], [1],
        [1, 1],
        [1],
        [1, 1],
        [1], [0, 0, 0],
        [1],
        [1], [1],
        [1, 1],
        [1],
        [1], [1], [0],
        [0, 0], [1, 1], [0, 0],
        [1], [0, 0],
        [1],
        [1], [0], [1], [0],
        [0], [1], [0], [1],
        [1],
        [0], [1],
        [0], [1],
        [1, 1],
        [1],
        [1, 1], [0, 0],
        [1],
        [1], [0, 0, 0],
        [1], [0, 0],
        [1], [1], [1],
        [0, 0], [1],
        [1],
        [0, 0], [1], [0, 0, 0],
        [1],
        [1],
        [1],
        [1], [1],
    ]

    return probs, positives

def load_jawfracnet_results():
    probs = [
        [0.8104], [0.8121],
        [0.7798], [0.7747],
        [0.8223], [0.7838],
        [-1],
        [0.8041],
        [0.8146],
        [0.8074],
        [0.8214], [0.6861],
        [0.8344],
        [0.8242], [0.8263],
        [0.7940],
        [0.8389],
        [0.8323], [0.8519],
        [0.7836],
        [0.8117],
        [0.8204],
        [0.7529], [0.8347],
        [0.8148], [0.5041],
        [0.8427],
        [0.8156],
        [0.7804],
        [0.7857],
        [0.8396],
        [0.7938],
        [0.8133],
        [0.8446],
        [0.7862],
        [0.8337], [0.8450], [0.8219],
        [0.8264],
        [0.8385],
        [0.8457],
        [0.6283],
        [0.8057],
        [0.5237],
        [0.6957], [-1]
    ]

    positives = [
        [1], [1],
        [1], [1],
        [1], [1],
        [1],
        [1],
        [1],
        [1],
        [1], [0],
        [1],
        [1], [1],
        [1],
        [1],
        [1], [1],
        [1],
        [1],
        [1],
        [1], [1],
        [1], [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1], [1], [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1], [1],
    ]

    return probs, positives

if __name__ == '__main__':
    probs, positives = load_jawfracnet_results()
    fps, recalls, score = compute_froc_curve(probs, positives, num_scans=35)

    # color = next(plt.gca()._get_lines.prop_cycler)['color']
    # first_color = color
    plt.plot(fps, recalls, label=f'JawFracNet ({score:.3f})', zorder=2)

    probs, positives = load_nnunet_results()
    fps, recalls, score = compute_froc_curve(probs, positives, num_scans=35)

    # color = next(plt.gca()._get_lines.prop_cycler)['color']
    # plt.plot(fps, recalls, label=f'nnU-Net ({score:.3f})', zorder=2)
    # plt.plot([0, 0], [-0.002, 0.5667], linestyle=(1, [5, 5]), c=color)
    # plt.plot([0.172, 0.6], [0.956, 0.956], linestyle=(6, [5, 5]), c=color)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.scatter([3 / 35], [44 / 45], label='R1', s=25, c=colors[2], zorder=3)
    # color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.scatter([5 / 35], [43 / 45], label='R2', s=25, c=colors[1], zorder=3)
    # color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.scatter([4 / 35], [36 / 45], label='R3', s=25, c=colors[3], zorder=3)

    plt.xlabel('False positives per scan')
    plt.ylabel('Sensitivity')
    plt.ylim(0, 1)
    # plt.title('FROC analysis')  
    plt.grid(zorder=0)
    plt.legend()

    plt.savefig('froc.png', dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()
