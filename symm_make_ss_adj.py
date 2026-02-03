import os,sys,glob,torch,random
import numpy as np
import argparse
import matplotlib.pyplot as plt

def ss_to_tensor(ss):
    """
    Function to convert ss files to indexed tensors
    0 = Helix
    1 = Strand
    2 = Loop
    3 = Mask/unknown
    """
    ss_conv = {'H':0.,'E':1.,'L':2.,'M':3.}
    ss_int = torch.tensor(np.array([(ss_conv[i]) for i in ss]), dtype=torch.float32)
    return ss_int

def construct_features(specs):
    with open(specs, 'r') as f:
        lines = f.readlines()
    features = {}
    feature_contacts = {}
    for l in lines:
        l = l.replace('\n', '').split(' ')
        if l[0].isnumeric():
            assert len(l) == 3, '!! Wrong format!! For each feature, you need to provide the following: ID, Secondary Structure, Length. Ex: "1 H 20"'
            assert (int(l[0]) not in features.keys()), '!! Each feature needs to have a unique ID !!'
            assert l[1] in 'HLEM', '!! Secondary structure not recognized !! Should be: H/E/L/M'
            assert l[2].isnumeric(), '!! Length needs to be specified for each feature !!'
            features[int(l[0])] = (l[1], int(l[2]))
        elif l[0].isalpha(): 
            if l[0].lower() == 'intra':
                contacts = l[1:]
                intra = []
                for c in contacts:
                    c = c.split(',')
                    assert len(c) == 2, f'!! Intra contacts {c} needs to be pairwise !!'
                    for j in c:
                        assert j.isnumeric(), '!! Not recognized ID for Intra contacts !!'
                        assert int(j) in features.keys(), '!! Not recognized ID for Intra contacts, make sure features are defined first !!'
                    c = (int(c[0]),int(c[1]))
                    intra.append(c)
                feature_contacts['intra'] = intra
            if l[0].lower() == 'inter':
                contacts = l[1:]
                inter = []
                for c in contacts:
                    c = c.split(',')
                    assert len(c) == 2, f'!! Inter contacts {c} needs to be pairwise !!'
                    for j in c:
                        assert j.isnumeric(), '!! Not recognized ID for Inter contacts !!'
                        assert int(j) in features.keys(), '!! Not recognized ID for Inter contacts, make sure features are defined first !!'
                    c = (int(c[0]),int(c[1]))
                    inter.append(c)
                feature_contacts['inter'] = inter
    return features, feature_contacts

def symm_ss_tensor(features, nsub):
    feature_list = sorted(list(features.keys()))
    ss = ''
    for f in feature_list:
        elem = features[f][0]*features[f][1]
        ss += elem
    ss *= nsub
    ss_int = ss_to_tensor(ss)
    return ss_int

def symm_block_adj(features, feature_contacts, nsub, ss_int):
    adj = torch.zeros([len(ss_int), len(ss_int)])
    feature_list = sorted(list(features.keys()))

    # # go through each self ss feature
    Lasu = sum([features[f][1] for f in feature_list])

    if 'intra' in feature_contacts.keys():
        contacts_list = feature_contacts['intra']
        for contact in contacts_list:
            for i in range(nsub):
                c1 = contact[0]
                c2 = contact[1]
                # calc position of c1 in ASU
                c1_start = sum([features[f][1] for f in feature_list[:feature_list.index(c1)]]) + Lasu*i
                c1_end = c1_start + features[c1][1]
                # calc position of c2 in ASU
                c2_start = sum([features[f][1] for f in feature_list[:feature_list.index(c2)]]) + Lasu*i
                c2_end = c2_start + features[c2][1]
                adj[c1_start:c1_end, c2_start:c2_end] = 1
                adj[c2_start:c2_end, c1_start:c1_end] = 1

    # go through each contact features, inter
    if 'inter' in feature_contacts.keys():
        contacts_list = feature_contacts['inter']
        for contact in contacts_list:
            for i in range(nsub-1):
                c1 = contact[0]
                c2 = contact[1]
                # calc position of c1 in ASU
                c1_start = sum([features[f][1] for f in feature_list[:feature_list.index(c1)]]) + Lasu*i
                c1_end = c1_start + features[c1][1]
                # calc position of c2 in ASU
                c2_start = sum([features[f][1] for f in feature_list[:feature_list.index(c2)]]) + Lasu*(i+1)
                c2_end = c2_start + features[c2][1]
                adj[c1_start:c1_end, c2_start:c2_end] = 1
                adj[c2_start:c2_end, c1_start:c1_end] = 1
    return adj, Lasu

def main(args):
    features, feature_contacts = construct_features(args.def_file)
    ss_int = symm_ss_tensor(features, args.nsub)
    adj, Lasu = symm_block_adj(features, feature_contacts, args.nsub, ss_int)
    outfolder = args.outfolder
    if outfolder[-1] != '/':
        outfolder = outfolder + '/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    prefix = outfolder.split('/')[-2]
    torch.save(adj, f'{outfolder}{prefix}_adj.pt')
    torch.save(ss_int, f'{outfolder}{prefix}_ss.pt')
    plt.imshow(adj, cmap='viridis', interpolation='nearest')
    for i in range(args.nsub-1):
        offset = Lasu - 1 + i*Lasu
        plt.axhline(y=offset, linestyle='-', linewidth=1, c='white')  # Horizontal line at y=4
        plt.axvline(x=offset, linestyle='-', linewidth=1, c='white')
    plt.colorbar()  # Add a color bar to show the scale
    plt.savefig(f'{outfolder}adj.png', dpi=600)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--def_file', type=str, default=None, help="Path to feature definition file")
    argparser.add_argument('--nsub', type=int, default=None, help="Number of modeled subunits for symmetry")
    argparser.add_argument('--outfolder', type=str, default=None, help='Folder where to save outputs')
    args = argparser.parse_args()    
    main(args)
