#!/usr/bin/env python

import os, sys
from collections import OrderedDict
import time
import argparse
import time
import torch

import util_protein_mpnn as mpnn_util

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'include'))

from pyrosetta import *
from pyrosetta.rosetta import *

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument( "-checkpoint_path", type=str, required=True, help='The path to the set of ProteinMPNN model weights that you would like to use' )
parser.add_argument( "-mpnn_cycles", type=int, default="10", help='mpnn sequences' )
parser.add_argument( "-relax_cycles", type=int, default="1", help="The number of ProteinMPNN->FastRelax cycles to perform (default 2)" )
parser.add_argument( "-output_intermediates", action="store_true", help='Whether to write all intermediate sequences from the relax cycles to disc (defaut False)' )
parser.add_argument( "-temperature", type=float, default=0.000001, help='The temperature to use for ProteinMPNN sampling (default 0)' )
parser.add_argument( "-augment_eps", type=float, default=0, help='The variance of random noise to add to the atomic coordinates (default 0)' )
parser.add_argument( "-omit_AAs", type=str, default='CX', help='A string off all residue types (one letter case-insensitive) that you would not like to use for design. Letters not corresponding to residue types will be ignored' )
parser.add_argument( "-num_connections", type=int, default=48, help='Number of neighbors each residue is connected to (default 48)' )
parser.add_argument( "-fix_FIXED_res", action="store_true", help='Whether to fix the sequence of residues labelled as FIXED or not (default False)' )
parser.add_argument( "-pdbs", type=str, required=True, help='Input .pdb file or file with list of pdbs' )
args = parser.parse_args( sys.argv[1:] )


init("-beta_nov16")

omit_AAs = [ letter for letter in args.omit_AAs.upper() if letter in list("ARNDCQEGHILKMFPSTWYVX") ]

rundir = os.path.dirname(os.path.realpath(__file__))

xml = rundir + "/RosettaFastRelaxUtil.xml"
objs = protocols.rosetta_scripts.XmlObjects.create_from_file( xml )

# Load the movers we will need

FastRelax = objs.get_mover( 'FastRelax' )

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string

def get_fixed_res(pose):
    fixed_res = []
    pdb_info = pose.pdb_info()
    endA = pose.split_by_chain()[1].size()
    for i in range(1,endA+1):
        if pdb_info.res_haslabel(i,"FIXED"):
            fixed_res.append(i)
    return fixed_res

def thread_mpnn_seq( pose, binder_seq ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    fixed = get_fixed_res(pose)

    for resi, mut_to in enumerate( binder_seq ):
        resi += 1 # 1 indexing

        if pose.residue(resi).name().split(':')[-1] != 'disulfide':
            name3 = aa_1_3[ mut_to ]
            new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )

            if resi not in fixed:
                pose.replace_residue( resi, new_res, True )
    return pose

def sequence_optimize( pdbfile, chains, model, fixed_positions_dict ):
    
    t0 = time.time()

    feature_dict = mpnn_util.generate_seqopt_features( pdbfile, chains )

    seq_per_struct = args.mpnn_cycles
    arg_dict = mpnn_util.set_default_args( seq_per_struct, omit_AAs=omit_AAs )
    arg_dict['temperature'] = args.temperature

    masked_chains = [ chains[0] ]
    visible_chains = [ chains[1] ]

    sequences = mpnn_util.generate_sequences( model, device, feature_dict, arg_dict, masked_chains, visible_chains, fixed_positions_dict )
    
    print( f"MPNN generated {len(sequences)} sequences in {int( time.time() - t0 )} seconds" ) 

    return sequences

def get_chains( pose ):
    lengths = [ p.size() for p in pose.split_by_chain() ]
    endA = pose.split_by_chain()[1].size()
    endB = endA + pose.split_by_chain()[2].size()

    chains = [ pose.pdb_info().chain( i ) for i in [ endA, endB ] ]

    return chains

def relax_pose( pose ):
    FastRelax.apply( pose )
    return pose

def get_fixed_res(pose):
    fixed_res = []
    pdb_info = pose.pdb_info()
    endA = pose.split_by_chain()[1].size()
    for i in range(1,endA+1):
        if pdb_info.res_haslabel(i,"FIXED"):
            fixed_res.append(i)
    return fixed_res

def mpnn_best( seqs ):
  bestseq, bestscore = seqs[0]
  for i in seqs[1:]:
    seq, score = i
    print(f'{score} {seq}')
    if score < bestscore:
      bestseq = seq
      bestscore = score
    print(f'best: {bestscore} {bestseq}')
  return bestseq, bestscore

def dl_design( pose, pdb, mpnn_model ):

    tot_t0 = time.time()

    tag = pdb.split('.pdb')[0]

    prefix = f"{tag}_dldesign"
    pdbfile = f"{prefix}_tmp.pdb"

    if args.fix_FIXED_res:
        fixed_res = get_fixed_res( pose )
    else:
        fixed_res = []

    fixed_positions_dict = None
    if len(fixed_res)>0:
        fixed_positions_dict = {}
        fixed_positions_dict[my_rstrip(pdbfile,'.pdb')] = {"A":fixed_res,"B":[]}
        print("Found residues with FIXED label, fixing the following residues: ", fixed_positions_dict[my_rstrip(pdbfile,'.pdb')])

    for cycle in range(args.relax_cycles):
        pose.dump_pdb( pdbfile )
        chains = get_chains( pose )
        seqs_scores = sequence_optimize( pdbfile, chains, mpnn_model, fixed_positions_dict )
        os.remove( pdbfile )
        seq, mpnn_score = mpnn_best(seqs_scores)
        pose = thread_mpnn_seq( pose, seq )
        pose = relax_pose(pose)
        if args.output_intermediates:
            tag = f"{prefix}_0_cycle{cycle}"
            pose.dump_pdb( tag+'.pdb' )

    # Do the final sequence assignment
    pose.dump_pdb( pdbfile )
    chains = get_chains( pose )
    seqs_scores = sequence_optimize( pdbfile, chains, mpnn_model, fixed_positions_dict )
    os.remove( pdbfile )
    
    if args.relax_cycles == 0:
        seqcnt = 1
        for i in seqs_scores:
            seq, score = i
            print(f'{score} {seq}')
            pose = thread_mpnn_seq( pose, seq )
            pose.dump_pdb(f"{prefix}_0_cycle{args.relax_cycles}_{seqcnt}.pdb")
            seqcnt += 1
    else:
        seq, mpnn_score = mpnn_best(seqs_scores)
        pose = thread_mpnn_seq( pose, seq )
        pose.dump_pdb(f"{prefix}_0_cycle{args.relax_cycles}.pdb")

# Checkpointing Functions

def record_checkpoint( tag, checkpoint_filename ):
    with open(checkpoint_filename, 'a') as f:
        f.write( tag )
        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
    done_set = set()
    if not os.path.isfile( checkpoint_filename ): return done_set

    with open( checkpoint_filename, 'r' ) as f:
        for line in f:
            done_set.add( line.strip() )

    return done_set

# End Checkpointing Functions

checkpoint_filename = "check.point_mpnn"
finished_structs = determine_finished_structs( checkpoint_filename )

def main( pdbs, mpnn_model ):
    for pdb in pdbs:
        t0 = time.time()
        print( "Attempting pose: %s"%pdb )
        ctag = pdb.split('.pdb')[0]
        if ctag in finished_structs:
            print( f"SKIPPING {ctag}, since it was already run" )
            continue

        pose = pose_from_file(pdb)
        dl_design( pose, pdb, mpnn_model )
        seconds = int(time.time() - t0)
        print( f"{pdb} reported success. Generated in {seconds} seconds" )
        record_checkpoint( ctag, checkpoint_filename )

#################################
# Begin Main Loop
#################################

debug = True

if torch.cuda.is_available():
    print('Found GPU will run MPNN on GPU')
    device = "cuda:0"
else:
    print('No GPU found, running MPNN on CPU')
    device = "cpu"

mpnn_model = mpnn_util.init_seq_optimize_model(device, hidden_dim=128, num_layers=3, backbone_noise=args.augment_eps, num_connections=args.num_connections, checkpoint_path=args.checkpoint_path )

pdbs = []
if args.pdbs.endswith('.pdb'):
  for i in args.pdbs.split(','):
    pdbs.append(i)
else:
  with open(args.pdbs) as f:
    for l in f:
      pdbs.append(l.strip().split()[0])

main( pdbs, mpnn_model )



