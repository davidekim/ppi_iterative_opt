import os,sys
import numpy as np
import glob
from pathlib import Path
import argparse

cwd = os.getcwd()

###### CONFIG ###########################################################################
script_path = os.path.dirname(os.path.abspath(__file__))

# Point this to an optional RF diffusion Apptainer
# https://apptainer.org/docs/user/main/index.html
rf_diffusion_container = f"{script_path}/containers/RF_diffusion.sif"

# Point this to your RF diffusion installation
# https://github.com/RosettaCommons/RFdiffusion
rf_diffusion = f"{script_path}/rf_diffusion/run_inference.py"

# Point this to an optional MPNN Apptainer
mpnn_fr_container = f"{script_path}/containers/mpnn_binder_design.sif"

# Point this to your ProteinMPNN installation
# https://github.com/dauparas/ProteinMPNN
mpnn_fr = f"{script_path}/mpnn_fr/dl_interface_design.py"
mpnn_fr_checkpoint = f"{script_path}/mpnn_fr/ProteinMPNN/soluble_model_weights/v_48_020.pt"

# Point this to your Rosetta installation
rosetta_scripts = f"{script_path}/rosetta/latest/bin/rosetta_scripts.hdf5.linuxgccrelease"

# Point this to an optional AF2 Apptainer
af2_container = f"{script_path}/containers/af2_binder_design.sif"

# Point this to your AF2 installation
af2 = f"{script_path}/af2_initial_guess/interfaceAF2predict.py"

#########################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--outputdirname', type=str, default='ppi_iterative_opt_output', help='Name of output directory.')
parser.add_argument('--cycles', type=int, default=10, help='Iterations.')
parser.add_argument('--partial_diffusions', type=int, default=20, help='Number of partial diffusions per iteration.')
parser.add_argument('--total_traj', type=int, default=10, help='Number of partial diffusion trajectory models to include.')
parser.add_argument('--mpnns', type=int, default=1, help='Number of MPNN sequence designs.')
parser.add_argument('--mpnn_relax_cycles', type=int, default=0, help='Number of mpnn Rosetta relax cycles.')
parser.add_argument('--disulfidize', type=bool, default=False, help='Include disulfide designs.')
parser.add_argument('--cleanup', type=bool, default=True, help='Cleanup PDBs to save disk space.')
parser.add_argument('--verbose', type=bool, default=True, help='Chatty output.')
parser.add_argument('--force', type=bool, default=False, help='Ignore checkpoints and force running from the beginning.')
parser.add_argument('pdbs', nargs=argparse.REMAINDER)
args = vars(parser.parse_args())
exit = False

outputdirname =  args['outputdirname']
cycles =  args['cycles']
partial_diffusions = args['partial_diffusions']
total_traj = args['total_traj']
mpnns = args['mpnns']
mpnn_relax_cycles = args['mpnn_relax_cycles']
disulfidize = args['disulfidize']
cleanup_pdbs = args['cleanup']
verbose = args['verbose']
force = args['force']

pdbs = []
if args['pdbs']:
  for pdb in args['pdbs']:
    if pdb.endswith('.pdb'):
      pdbs.append(pdb)
    else:
      with open(pdb) as f:
        for l in f:
          pdb = l.strip().split()[0]
          if pdb.endswith('.pdb'):
            pdbs.append(pdb)
else:
  exit = True
if exit:
  parser.print_help(sys.stderr)
  sys.exit(1)

if verbose:
  print(f'Optimizing: {pdb} with {cycles} cycles....')

ca_atoms = []
def chain1_len():
  chainAlen = 0
  prevchain = ''
  for ca in ca_atoms:
    if len(prevchain) and ca[0] != prevchain:
      break
    chainAlen += 1
    prevchain = ca[0]
  return chainAlen

def dist(xyz1, xyz2):
  nxyz1 = np.array(xyz1)
  nxyz2 = np.array(xyz2)  
  return np.linalg.norm(nxyz1 - nxyz2)

def read_pdb_atom(l):
  chain = l[20:22].strip()
  atype = l[11:17].strip()
  name3 = l[17:20].strip()
  resnum = int(l[22:26].strip())
  x = float(l[30:38])
  y = float(l[38:46])
  z = float(l[46:54])
  return chain, atype, name3, resnum, x, y, z

def chain2_contigs():
  chain1len = chain1_len()
  start =  ca_atoms[chain1len][0]+str(ca_atoms[chain1len][3])
  target_chain_breaks = []
  for i in range(chain1len,len(ca_atoms)-1):
    d = dist(ca_atoms[i][4:7],ca_atoms[i+1][4:7])
    if d > 4.2: # chainbreak
      target_chain_breaks.append(f'{start}-{ca_atoms[i][3]}')
      start = ca_atoms[i+1][0]+str(ca_atoms[i+1][3])
  target_chain_breaks.append(f'{start}-{ca_atoms[-1][3]}')
  return ','.join(target_chain_breaks)

def extract_traj(traj,prefix,s,n):
  with open(traj) as f:
    cnt = 1
    coords = []
    for l in f:
      if l.startswith('ENDMDL'):
        with open(f'{prefix}_{s}_traj{cnt}.pdb', 'w') as nf:
          for c in coords:
            nf.write(c)
        coords = []
        if cnt >= n:
          break
        cnt += 1
        continue
      coords.append(l)

def add_FIXED_disulfides(disulfidized):
  newlines = []
  fixed = []
  ignore = []
  with open(disulfidized) as f:
    for l in f:
      newlines.append(l)
      if l.startswith('SSBOND'):
        cols = l.split()
        if cols[2] == 'A' and cols[5] == 'A':
          fixed.append(cols[3])
          fixed.append(cols[6])
      if l.startswith("REMARK PDBinfo-LABEL"):
        cols = l.split()
        if cols[3] == 'FIXED':
          ignore.append(cols[2])
  with open(disulfidized+'_tmp', 'w') as f:
    for l in newlines:
      f.write(l)
    for pos in fixed:
      if pos not in ignore:
        f.write(f'REMARK PDBinfo-LABEL:{int(pos):5d} FIXED'+"\n") 
  os.replace(disulfidized+'_tmp', disulfidized)

for pdb in pdbs:
  ca_atoms = []
  with open(pdb) as f:
    for l in f:
      if l.startswith('ATOM'):
        chain, atype, name3, resnum, x, y, z = read_pdb_atom(l)
        if atype == 'CA':
          ca_atoms.append([chain, atype, name3, resnum, x, y, z])

  contigstr = f'{chain1_len()},0\\ {chain2_contigs()}'

  startpdb = pdb
  startpae = 9999999.9
  for n in range(1, cycles+1):
    if verbose:
      print(f'Cycle {n} {startpdb}....') 

    # PARTIAL DIFFUSION
    prefix = f'{cwd}/{outputdirname}/'+pdb.split('/')[-1].split('.pdb')[0]+f'_pd_cycle{n:05d}'
    if force or not os.path.exists(prefix+'_pddone'):
      cmd = f'{rf_diffusion_container} {rf_diffusion} inference.output_prefix={prefix} '
      cmd += f'inference.input_pdb={startpdb} contigmap.contigs=[\\\'{contigstr}\\\'] inference.num_designs={partial_diffusions} denoiser.noise_scale_ca=0.5 denoiser.noise_scale_frame=0.5 diffuser.partial_T=15'
      if verbose:
        print(f'running partial diffusion: {cmd}')
      if not os.system(cmd):
        for i in range(0,partial_diffusions):
          traj = f'{cwd}/{outputdirname}/traj/'+pdb.split('/')[-1].split('.pdb')[0]+f'_pd_cycle{n:05d}_{i}_pX0_traj.pdb'
          extract_traj(traj,prefix,i,total_traj)
        Path(prefix+'_pddone').touch()

    diffused = []
    for i in range(0,partial_diffusions): 
      diffused.append(f'{prefix}_{i}.pdb')
      for j in range(1,total_traj+1):
        diffused.append(f'{prefix}_{i}_traj{j}.pdb')
  
    # MPNN
    if force or not os.path.exists(prefix+'_mpnndone'):
      cmd = f' {mpnn_fr_container} {mpnn_fr} -checkpoint_path {mpnn_fr_checkpoint} '
      cmd += f' -mpnn_cycles {mpnns} -temperature 0.0001 -augment_eps 0 -relax_cycles {mpnn_relax_cycles} -pdb '+','.join(diffused)
      if verbose:
        print(f'running mpnn+relax: {cmd}')
      if not os.system(cmd):
        Path(prefix+'_mpnndone').touch()

    # Include Disulfidized?
    if disulfidize and (force or not os.path.exists(prefix+'_disulfidizedone1')):
      mpnned = []
      for i in glob.glob(f'{prefix}*cycle*.pdb'):
        if not i.endswith('af2pred.pdb') and not i.endswith('_0001.pdb'):
          mpnned.append(i)
      cmd = f'{rosetta_scripts} -parser:protocol {script_path}/disulfidize.xml '
      cmd += f'-corrections::beta_nov16 -out:path:all {cwd}/{outputdirname}/ -in:file:s '+' '.join(mpnned)
      if verbose:
        print(f'running disulfidize1: {cmd}')
      if not os.system(cmd):
        Path(prefix+'_disulfidizedone1').touch()

    # MPNN disulfidized (keeps disulfides fixed)
    if disulfidize and (force or not os.path.exists(prefix+'_disulfidizedone2')):
      disulfidized = []
      for i in glob.glob(f'{prefix}*cycle*.pdb'):
        if i.endswith('_0001.pdb'):
          add_FIXED_disulfides(i)
          disulfidized.append(i)
      cmd = f' {mpnn_fr_container} {mpnn_fr} -checkpoint_path {mpnn_fr_checkpoint} '
      cmd += f' -mpnn_cycles {mpnns} -temperature 0.0001 -augment_eps 0 -fix_FIXED_res -relax_cycles {mpnn_relax_cycles} -pdb '+','.join(disulfidized)
      if verbose:
        print(f'running mpnn+relax disulfidize2: {cmd}')
      if not os.system(cmd):
        Path(prefix+'_disulfidizedone2').touch()

    # AF2
    if force or (not os.path.exists(prefix+'_af2done') or not os.path.exists(prefix+'_af2.sc')):
      mpnned = []
      for i in glob.glob(f'{prefix}*cycle*.pdb'):
        if not i.endswith('af2pred.pdb') and not i.endswith('_0001.pdb'):
          mpnned.append(i)
      cmd = f'{af2_container} {af2} -scorefile {prefix}_af2.sc -pdbs '+','.join(mpnned)
      if verbose:
        print(f'running af2: {cmd}')
      if not os.system(cmd):
        Path(prefix+'_af2done').touch()

    # cleanup pdbs
    if cleanup_pdbs:
      for i in glob.glob(f'{prefix}*.pdb'):
        if not i.endswith('_af2pred.pdb'):
          os.remove(i)

    # Get pae_interaction scoring AF2
    top_design = ""
    top_pae = 99999.9
    with open(f'{prefix}_af2.sc') as f:
      header = {}
      for l in f:
        cols = l.strip().split()
        if len(cols) < 3:
          continue
        if len(header) == 0:
          for i,val in enumerate(cols):
            header[val] = i
          continue
        else:
          if l.find('pae_interaction') > -1:
            continue
          pae = float(cols[header['pae_interaction']])
          if pae < top_pae:
            top_pae = pae
            top_design = cols[header['description']]+'.pdb'

    if verbose:
      print(f'Cycle {n} top design: {top_design} pae_interaction: {top_pae}')
    if top_pae < 15 and top_pae < startpae:
      startpdb = top_design
      startpae = top_pae

    if n == 2 and startpae <= 4.0: # terminate run early, good enough!
      if verbose: print(f'Terminating run since iteration {n} pae {startpae} is <= 4')
      exit(0)

    if n == 3 and startpae <= 4.5: # terminate run early, good enough!
      if verbose: print(f'Terminating run since iteration {n} pae {startpae} is <= 4.5')
      exit(0)

    if n == 4 and startpae <= 4.5: # terminate run early, good enough!
      if verbose: print(f'Terminating run since iteration {n} pae {startpae} is <= 4.5')
      exit(0)
  
    if n == 6 and startpae >= 15: # terminate run early, horrible!
      if verbose: print(f'Terminating run since iteration {n} pae {startpae} has not reached pae < 15')
      exit(0)





