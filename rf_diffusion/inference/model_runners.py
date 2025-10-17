import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from icecream import ic
from RoseTTAFoldModel import RoseTTAFoldModule
from kinematics import get_init_xyz, xyz_to_t2d
from diffusion import Diffuser
import seq_diffusion
from chemical import seq2chars, INIT_CRDS
from util_module import ComputeAllAtomCoords
from contigs import ContigMap
from inference import utils as iu
from potentials.manager import PotentialManager
from inference import symmetry
import logging
import torch.nn.functional as nn
import util
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import pickle
import random
import sys
sys.path.append('../') # to access RF structure prediction stuff 

script_path = os.path.dirname(os.path.abspath(__file__))

# When you import this it causes a circular import due to the changes made in apply masks for self conditioning
# This import is only used for SeqToStr Sampling though so can be fixed later - NRB
# import data_loader 
from model_input_logger import pickle_function_call

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles

class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)
    
    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_override_path != self._conf.inference.ckpt_override_path

        # Assign config to Sampler
        self._conf = conf

        self.contig_conf = self._conf.contigmap
        # specific contigs
        # This is where a pickled file of contig inputs is provided.
        # Have to process this here to find the appropriate model
        seqonly_model_from_specific=False
        if self.contig_conf.specific_contig_pkl is not None:
            with open(self.contig_conf.specific_contig_pkl,'rb') as f:
                self.specific_contigs = pickle.load(f)
            if any('inpaint_seq_tensor' in x for x in  self.specific_contigs.items()):
                assert all('inpaint_seq_tensor' in x for x in self.specific_contigs.items()), "can't mix and match with providing inpaint_seq_tensor in the specific contigs"
                seqonly_model_from_specific=True
            if any('inpaint_str_tensor' in x for x in self.specific_contigs.items()):
                assert all('inpaint_str_tensor' in x for x in self.specific_contigs.items()), "can't mix and match with providing inpaint_str_tensor in the specific contigs"
                seqonly_model_from_specific=True 
        else:
            self.specific_contigs={}

        # Initialize inference only helper objects to Sampler
        # JW now added automatic model selection.
        if conf.inference.ckpt_override_path is not None:
            self.ckpt_path = conf.inference.ckpt_override_path
            print("WARNING: You're overriding the checkpoint path from the defaults. Check that the model you're providing can run with the inputs you're providing.")
        else:
            if conf.contigmap.inpaint_seq is not None or conf.contigmap.provide_seq is not None or conf.contigmap.inpaint_str is not None or seqonly_model_from_specific:
                # use model trained for inpaint_seq
                if conf.contigmap.provide_seq is not None:
                    # this is only used for partial diffusion
                    assert conf.diffuser.partial_T is not None, "The provide_seq input is specifically for partial diffusion"
                if conf.scaffoldguided.scaffoldguided:
                    self.ckpt_path=f'{script_path}/../models/seq_alone_models_FoldConditioned_Jan23/BFF_4.pt'
                else:
                    self.ckpt_path = f'{script_path}/../models/seq_alone_models_Dec2022/BFF_6.pt'
            elif conf.ppi.hotspot_res is not None and conf.scaffoldguided.scaffoldguided is False:
                # use complex trained model
                self.ckpt_path = f'{script_path}/../models/hotspot_models/base_complex_finetuned_BFF_9.pt'
            elif conf.scaffoldguided.scaffoldguided is True:
                # use complex and secondary structure-guided model
                self.ckpt_path = f'{script_path}/../models/hotspot_models/base_complex_ss_finetuned_BFF_9.pt' 
            else:
                # use default model
                self.ckpt_path = f'{script_path}/../models/BFF_4.pt'
        # for saving in trb file:
        assert self._conf.inference.trb_save_ckpt_path is None, "trb_save_ckpt_path is not the place to specify an input model. Specify in inference.ckpt_override_path"
        self._conf['inference']['trb_save_ckpt_path']=self.ckpt_path

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            self.assemble_config_from_chk()
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

        # self.initialize_sampler(conf)
        self.initialized=True

        # Assemble config from the checkpoint
        print(' ')
        print('-'*100)
        print(' ')
        print("WARNING: The following options are not currently implemented at inference. Decide if this matters.")
        print("Delete these in inference/model_runners.py once they are implemented/once you decide they are not required for inference -- JW")
        print(" -predict_previous")
        print(" -prob_self_cond")
        print(" -seqdiff_b0")
        print(" -seqdiff_bT")
        print(" -seqdiff_schedule_type")
        print(" -seqdiff")
        print(" -freeze_track_motif")
        print(" -use_motif_timestep")
        print(" ")
        print("-"*100)
        print(" ")
        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.diffuser = Diffuser(**self._conf.diffuser)
        # TODO: Add symmetrization RMSD check here
        if self._conf.seq_diffuser.seqdiff is None:
            ic('Doing AR Sequence Decoding')
            self.seq_diffuser = None

            assert(self._conf.preprocess.seq_self_cond is False), 'AR decoding does not make sense with sequence self cond'
            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        elif self._conf.seq_diffuser.seqdiff == 'continuous':
            ic('Doing Continuous Bit Diffusion')

            kwargs = {
                     'T': self._conf.diffuser.T,
                     's_b0': self._conf.seq_diffuser.s_b0,
                     's_bT': self._conf.seq_diffuser.s_bT,
                     'schedule_type': self._conf.seq_diffuser.schedule_type,
                     'loss_type': self._conf.seq_diffuser.loss_type
                     }
            self.seq_diffuser = seq_diffusion.ContinuousSeqDiffuser(**kwargs)

            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        else:
            sys.exit(f'Seq Diffuser of type: {self._conf.seq_diffuser.seqdiff} is not known')

        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.symmetry = None

        self.allatom = ComputeAllAtomCoords().to(self.device)
        
        if self.inf_conf.input_pdb is None:
            # set default pdb
            script_dir=os.path.dirname(os.path.realpath(__file__))
            self.inf_conf.input_pdb=os.path.join(script_dir, '../benchmark/input/1qys.pdb')
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
        self.chain_idx = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)

    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)

    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
        
        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

        JW
        """
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = list(HydraConfig.get().overrides.task)
            ic(overrides)
        # Added to set default T to 50
        overrides.append(f'diffuser.T={self._conf.diffuser.T}') if not any(i.startswith('diffuser.T=') for i in overrides) else None
        if 'config_dict' in self.ckpt.keys():
            print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

            # First, check all flags in the checkpoint config dict are in the config file
            for cat in ['model','diffuser','seq_diffuser','preprocess']:
                #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
                for key in self._conf[cat]:
                    if key == 'chi_type': # and self.ckpt['config_dict'][cat][key] == 'circular':
                        ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                        continue
                    try:
                        print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                        self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                    except:
                        print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {self._conf[cat][key]} is correct')
            # add back in overrides again
            for override in overrides:
                if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                    print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                    mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
        else:
            print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        
        # Now read input dimensions from checkpoint.
        self.d_t1d=self._conf.preprocess.d_t1d
        self.d_t2d=self._conf.preprocess.d_t2d
        model = RoseTTAFoldModule(**self._conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, T=self._conf.diffuser.T).to(self.device)
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference')
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def construct_contig(self, target_feats):
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        # Here we're providing specific contig inputs
        if len(self.specific_contigs.keys()) > 0:
            assert all(x is None for x in [self.contig_conf.contigs, self.contig_conf.inpaint_str, self.contig_conf.inpaint_seq, self.contig_conf.length, self.contig_conf.provide_seq]), "If providing specific contigs you can't also be providing contig arguments at the command line" 
            specific_contig=self.specific_contigs[random.choice(list(self.specific_contigs.keys()))]
            specific_contig=util.fix_dtypes(specific_contig)
            return ContigMap(target_feats, **self.contig_conf, **specific_contig)
        return ContigMap(target_feats, **self.contig_conf)
    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        # TODO: Denoiser seems redundant. Combine with diffuser.
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        aa_decode_steps = min(denoise_kwargs['aa_decode_steps'], denoise_kwargs['partial_T'] or 999)
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'seq_diffuser': self.seq_diffuser,
            'potential_manager': self.potential_manager,
            'visible': visible,
            'aa_decode_steps': aa_decode_steps,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """
        # process target and reinitialise potential_manager. This is here because the 'target' is always set up to be the second chain in out inputs. Could change this down the line
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)

        # moved this here as should be updated each iteration of diffusion
        self.contig_map = self.construct_contig(self.target_feats)
        self.mappings = self.contig_map.get_mappings()
        self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None,:]
        self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None,:]
        self.binderlen =  len(self.contig_map.inpaint)        
        self.hotspot_0idx=iu.get_idx0_hotspots(self.mappings, self.ppi_conf, self.binderlen) 
        # Now initialise potential manager here. This allows variable-length binder design
        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.hotspot_0idx,
                                                  self.binderlen)
        target_feats = self.target_feats
        contig_map = self.contig_map

        xyz_27 = target_feats['xyz_27']
        mask_27 = target_feats['mask_27']
        seq_orig = target_feats['seq']
        L_mapped = len(self.contig_map.ref)
        
        diffusion_mask = self.mask_str
        self.diffusion_mask = diffusion_mask
        
        if not self.contig_map.specific_mappings: 
            self.chain_idx=['A' if i < self.binderlen else 'B' for i in range(L_mapped)]
        else:
            self.chain_idx=[i[0] for i in self.contig_map.hal]
            self.out_idx=[int(i[1]) for i in self.contig_map.hal]
        # adjust size of input xt according to residue map 
        if self.diffuser_conf.partial_T:
            assert xyz_27.shape[0] == L_mapped, f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {xyz_27.shape[0]} != {L_mapped}"
            assert contig_map.hal_idx0 == contig_map.ref_idx0, f'for partial diffusion there can be no offset between the index of a residue in the input and the index of the residue in the output, {contig_map.hal_idx0} != {contig_map.ref_idx0}'
            # Partially diffusing from a known structure
            xyz_mapped=xyz_27
            atom_mask_mapped = mask_27
        else:
            # Fully diffusing from points initialised at the origin
            # adjust size of input xt according to residue map
            xyz_mapped = torch.full((1,1,L_mapped,27,3), np.nan)
            xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0,...]
            xyz_motif_prealign = xyz_mapped.clone()
            motif_prealign_com = xyz_motif_prealign[0,0,:,1].mean(dim=0)
            self.motif_com = xyz_27[contig_map.ref_idx0,1].mean(dim=0)
            xyz_mapped = get_init_xyz(xyz_mapped).squeeze()
            # adjust the size of the input atom map
            atom_mask_mapped = torch.full((L_mapped, 27), False)
            atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input+1)

        # Moved this here so that the sequence diffuser has access to t_list
        # NOTE: This is where we switch from an integer sequence to a one-hot sequence - NRB
        if self.seq_diffuser is None:
            seq_t = torch.full((1,L_mapped), 21).squeeze()
            seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
            seq_t[~self.mask_seq.squeeze()] = 21

            seq_t    = torch.nn.functional.one_hot(seq_t, num_classes=22).float() # [L,22]
            seq_orig = torch.nn.functional.one_hot(seq_orig, num_classes=22).float() # [L,22]
        else:
            # Sequence diffusion
            # Noise sequence using seq diffuser
            seq_mapped = torch.full((1,L_mapped), 0).squeeze()
            seq_mapped[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]

            diffused_seq_stack, seq_orig = self.seq_diffuser.diffuse_sequence( 
                    seq = seq_mapped,
                    diffusion_mask = self.mask_seq.squeeze(),
                    t_list = t_list
                    )

            seq_t = torch.clone(diffused_seq_stack[-1]) # [L,20]

            zeros = torch.zeros(L_mapped,2)
            seq_t = torch.cat((seq_t,zeros), dim=-1) # [L,22]

        fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
            xyz_mapped,
            torch.clone(seq_t),  # TODO: Check if copy is needed.
            atom_mask_mapped.squeeze(),
            diffusion_mask=diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)
        xT = fa_stack[-1].squeeze()[:,:14,:]
        xt = torch.clone(xT)

        if self.diffuser_conf.partial_T and self.seq_diffuser is None:
            is_motif = self.mask_seq.squeeze()
            is_shown_at_t = torch.tensor(aa_masks[-1])
            visible = is_motif | is_shown_at_t
            if self.diffuser_conf.partial_T:
                seq_t[visible] = seq_orig[visible]
        else:
            # Sequence diffusion
            visible = self.mask_seq.squeeze()

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=visible)
        if self.symmetry is not None:
            xt, seq_t = self.symmetry.apply_symmetry(xt, seq_t)
        self._log.info(f'Sequence init: {seq2chars(torch.argmax(seq_t, dim=-1))}')
        
        if return_forward_trajectory:
            forward_traj = torch.cat([xyz_true[None], fa_stack[:,:,:]])
            if self.seq_diffuser is None:
                aa_masks[:, diffusion_mask.squeeze()] = True
                return xt, seq_t, forward_traj, aa_masks, seq_orig
            else:
                # Seq Diffusion
                return xt, seq_t, forward_traj, diffused_seq_stack, seq_orig
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None
        # For the implicit ligand potential
        if self.potential_conf.guiding_potentials is not None:
            if any(list(filter(lambda x: "substrate_contacts" in x, self.potential_conf.guiding_potentials))):
                assert len(self.target_feats['xyz_het']) > 0, "If you're using the Substrate Contact potential, you need to make sure there's a ligand in the input_pdb file!"
                het_names = np.array([i['name'].strip() for i in self.target_feats['info_het']])
                xyz_het = self.target_feats['xyz_het'][het_names == self._conf.potentials.substrate]
                xyz_het = torch.from_numpy(xyz_het)
                assert xyz_het.shape[0] > 0, f'expected >0 heteroatoms from ligand with name {self._conf.potentials.substrate}'
                xyz_motif_prealign = xyz_motif_prealign[0,0][self.diffusion_mask.squeeze()]
                motif_prealign_com = xyz_motif_prealign[:,1].mean(dim=0)
                xyz_het_com = xyz_het.mean(dim=0)
                for pot in self.potential_manager.potentials_to_apply: # fix this
                    pot.motif_substrate_atoms = xyz_het
                    pot.diffusion_mask = self.diffusion_mask.squeeze()
                    pot.xyz_motif = xyz_motif_prealign
                    pot.diffuser = self.diffuser
        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t, repack=False):
        
        """
        Function to prepare inputs to diffusion model
        
            seq (L,22) one-hot sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)
        
            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)
                - contacting residues: for ppi. Target residues in contact with biner (1)
                - chi_angle timestep (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (1, L, L, 45)
                - last plane is block adjacency
    """

        L = seq.shape[0]
        T = self.T
        binderlen = self.binderlen
        target_res = self.ppi_conf.hotspot_res

        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,48))
        msa_masked[:,:,:,:22] = seq[None, None]
        msa_masked[:,:,:,22:44] = seq[None, None]
        if self._conf.inference.annotate_termini:
            msa_masked[:,:,0,46] = 1.0
            msa_masked[:,:,-1,47] = 1.0

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,25))
        msa_full[:,:,:,:22] = seq[None, None]
        if self._conf.inference.annotate_termini:
            msa_full[:,:,0,23] = 1.0
            msa_full[:,:,-1,24] = 1.0

        ### t1d ###
        ########### 
        # NOTE: Not adjusting t1d last dim (confidence) from sequence mask

        # Here we need to go from one hot with 22 classes to one hot with 21 classes
        t1d = torch.zeros((1,1,L,21))

        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx,21] == 1:
                seqt1d[idx,20] = 1
                seqt1d[idx,21] = 0
        
        t1d[:,:,:,:21] = seqt1d[None,None,:,:21]
        
        # Str Confidence
        if self.inf_conf.autoregressive_confidence:
            # Set confidence to 1 where diffusion mask is True, else 1-t/T
            strconf = torch.zeros((L)).float()
            strconf[self.mask_str.squeeze()] = 1.
            strconf[~self.mask_str.squeeze()] = 1. - t/self.T
            strconf = strconf[None,None,...,None]
        else:
            #NOTE: DJ - I don't know what this does or why it's here
            strconf = torch.where(self.mask_str.squeeze(), 1., 0.)[None,None,...,None]
        t1d = torch.cat((t1d, strconf), dim=-1)
        
        # Seq Confidence
        if self.inf_conf.autoregressive_confidence:
            # Set confidence to 1 where diffusion mask is True, else 1-t/T
            seqconf = torch.zeros((L)).float()
            seqconf[self.mask_seq.squeeze()] = 1.
            seqconf[~self.mask_seq.squeeze()] = 1. - t/self.T
            seqconf = seqconf[None,None,...,None]
        else:
            #NOTE: DJ - I don't know what this does or why it's here
            seqconf = torch.where(self.mask_seq.squeeze(), 1., 0.)[None,None,...,None]
        # Seqdiff confidence is added as 23rd dimension 
        if self.seq_diffuser is not None:
            t1d = torch.cat((t1d, seqconf), dim=-1)
        t1d = t1d.float()
        
        ### xyz_t ###
        #############
        if self.preprocess_conf.sidechain_input:
            xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
        else:
            xyz_t[~self.mask_str.squeeze(),3:,:] = float('nan')
        #xyz_t[:,3:,:] = float('nan')

        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)

        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t)
        
        ### idx ###
        ###########
        """
        idx = torch.arange(L)[None]
        if ppi_design:
            idx[:,binderlen:] += 200
        """
        # JW Just get this from the contig_mapper now. This handles chain breaks
        idx = torch.tensor(self.contig_map.rf)[None]

        ### alpha_t ###
        ###############
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        #put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        
        ### added_features ###
        ######################
        # NB the hotspot input has been removed in this branch. 
        # JW added it back in, using pdb indexing
        if self.preprocess_conf.d_t1d >= 24: # add hotpot residues
            hotspot_tens = torch.zeros(L).float()
            if self.ppi_conf.hotspot_res is None:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots. If you're doing monomer diffusion this is fine")
                hotspot_idx=[]
            else:
                hotspots = [(i[0],int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx=[]
                for i,res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
                hotspot_tens[hotspot_idx] = 1.0

            # NB penultimate plane relates to sequence self conditioning. In these models set it to zero.
            if self.seq_diffuser is None:
                t1d=torch.cat((t1d, torch.zeros_like(t1d[...,:1]), hotspot_tens[None,None,...,None].to(self.device)), dim=-1)
            else:
                # already concatenated on the seq conf dimension
                t1d=torch.cat((t1d, hotspot_tens[None,None,...,None].to(self.device)), dim=-1)
        """
        # t1d
        if self.preprocess_conf.d_t1d == 23: # add hotspot residues
            # NRB: Adding in dimension for target hotspot residues
            target_residue_feat = torch.zeros_like(t1d[...,0])[...,None]
            if ppi_design and not target_res is None:
                absolute_idx = [resi+binderlen for resi in target_res]
                target_residue_feat[...,absolute_idx,:] = 1
            t1d = torch.cat((t1d, target_residue_feat), dim=-1)
            t1d = t1d.float()
        """ 
        return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        
    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step, return_extra=False):
        '''Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
            
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L,22) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        out = self._preprocess(seq_t, x_t, t)
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)

        N,L = msa_masked.shape[:2]

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        # decide whether to recycle information between timesteps or not
        if self.inf_conf.recycle_between and t < self.diffuser_conf.aa_decode_steps:
            msa_prev = self.msa_prev
            pair_prev = self.pair_prev
            state_prev = self.state_prev
        else:
            msa_prev = None
            pair_prev = None
            state_prev = None

        with torch.no_grad():
            # So recycling is done a la training
            px0=xt_in
            for _ in range(self.recycle_schedule[t-1]):
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = msa_prev,
                                    pair_prev = pair_prev,
                                    state_prev = state_prev,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))

        self.msa_prev=msa_prev
        self.pair_prev=pair_prev
        self.state_prev=state_prev
        # prediction of X0 
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]
        #sampled_seq = torch.argmax(logits.squeeze(), dim=-1)
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position 
        
        # grab only the query sequence prediction - adjustment for Seq2StrSampler
        sampled_seq = sampled_seq.reshape(N,L,-1)[0,0]

        # Process outputs.
        mask_seq = self.mask_seq

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device)

        pseq_0[mask_seq.squeeze()] = seq_init[
            mask_seq.squeeze()].to(self.device)

        seq_t = torch.nn.functional.one_hot(
            seq_t, num_classes=22).to(self.device)

        self._log.info(
           f'Timestep {t}, current sequence: { seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')
        
        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_diffusion_mask=self.mask_seq.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = torch.clone(pseq_0)
            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)
        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
        if return_extra:
            return px0, x_t_1, seq_t_1, tors_t_1, plddt, logits
        return px0, x_t_1, seq_t_1, tors_t_1, plddt

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output, either for recycling or for self-conditioning
        """
        _,px0_aa = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym,_ = self.symmetry.apply_symmetry(px0_aa.to('cpu').squeeze()[:,:14], torch.argmax(seq_in, dim=-1).squeeze().to('cpu'))
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym

    def should_we_run_rf2( self, t, T, partial_T ):
        if self.inf_conf.run_rf2_at_t is None:
            return True

        # There's ast.eval(), but you'd have to string replace t, which doesn't feel right either. (Thing "not")
        should_run = eval( self.inf_conf.run_rf2_at_t.replace("_", " ") )

        if t == T or t == partial_T:
            assert should_run, "inference.run_rf2_at_t set to not allow RF2 on first timestep {t}. At present, this results in an error"

        return should_run

class Seq2StrSampler(Sampler):
    """
    Model runner for RF_diffusion fixed sequence structure prediction with or w/o MSA
    """
    
    def __init__(self, conf: DictConfig):
        super().__init__(conf)
        raise NotImplementedError("Seq2Str Sampler not implemented yet")
        self.seq2str_conf = conf.seq2str

        # parse MSA from config
        msa, ins   = iu.parse_a3m(self.seq2str_conf.input_a3m) # msa - (N,L) integers  
        self.msa   = torch.from_numpy(msa).long()
        self.query = torch.from_numpy(msa[0]).long()
        self.ins   = torch.from_numpy(ins).long()

        self.mask_seq = torch.zeros_like(self.query).bool()
        self.mask_str = self.mask_seq.clone()

        self.L = msa.shape[-1]

        # if complex modelling, get chain lengths 
        # Assumes chains are in same order as sequence 
        if self.seq2str_conf.chain_lengths:
            self.chain_lengths = [int(a) for a in self.seq2str_conf.chain_lengths.split(',')] 
        else:
            self.chain_lengths = [self.L]
        assert sum(self.chain_lengths) == self.L, 'input chain lengths did not sum to query sequence length'
        
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
    
    def sample_init(self): 
        """
        Create an initialized set of residues to go through the model
        """
        # initial structure is completely unknown
        # Diffuser will take care of filling in the atoms when protein is diffused for the first time 
        x_nan = torch.full((self.L, 27,3), np.nan)
        x_nan = get_init_xyz(x_nan[None,None]).squeeze()
        atom_mask = torch.full((self.L,27), False)
        
        seq_T = self.query # query sequence
        self.diffusion_mask = torch.full((self.L,),False)

        # Setup denoiser
        self.denoiser = self.construct_denoiser(self.L)

        fa_stack,_,_,_ = self.diffuser.diffuse_pose(
            x_nan,
            torch.clone(seq_T),  # TODO: Check if copy is needed.
            atom_mask.squeeze(),
            diffusion_mask=self.diffusion_mask,
            t_list=[self.diffuser_conf.T],
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)
        
        # the most diffused set of atoms is the last one 
        xT = torch.clone(fa_stack[-1].squeeze()[:,:14,:]) 
        
        # from the outside it's returned as seq_T
        return xT, seq_T


    def _preprocess(self, seq_t, xyz_t, t):

        """
        Function to prepare inputs to diffusion model - but now with MSA + MSA statistics etc 
        
            seq (L) integer sequence 
        """
        msa = self.msa 
        ins = self.ins
        N,L = msa.shape        

        #### Build template features #### 

        # rename just so we dont get confused - template is indeed at timestep t 
        xyz_template = xyz_t

        #### Build MSA features #### 
        params = {
        "MINTPLT" : 0,
        "MAXTPLT" : 5,
        "MINSEQ"  : 1,
        "MAXSEQ"  : 1024,
        "MAXLAT"  : 128,
        "CROP"    : 256,
        "BLOCKCUT": 5,
        "ROWS"    : 1,
        "SEQID"   : 95.0,
        "MAXCYCLE": 1,
        }
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = data_loader.MSAFeaturize(msa, ins, params)
        msa_masked = msa_seed
        msa_full   = msa_extra 

        
        ### t1d ### 
        ########### 
        # seq 21, conf 1,
        t1d = torch.zeros((1,self.L, 22+9))

        # seq query sequence one hot encoding
        t1d[:,:,self.query] = 1  

        # First template (the diffused) has zero confidence 
        t1d[:1,:,21] = 0
        
        # SS 
        # all SS features are set to masked (26th element, 1-hot)
        t1d[:1,:,25] = 1 

        # Add global timestep. 
        t1d[:1,:,26] = 1-t/self.diffuser.T

        # Add chi angle timestep, same as global timestep
        t1d[:1,:,27] = 1-t/self.diffuser.T 

        # Add contacting residues.
        t1d[:1,:,28] = torch.zeros(1,L)

        # Add diffused or not (1 = not, 0 = diffused)
        t1d[:1,:,29] = torch.zeros(1,L)

        # Feature indicating whether this is a real homologous template 
        # or just the diffused input 
        # (1=template, 0=diffused)
        t1d[:1,:,30] = torch.zeros(1,L)

        t1d = t1d[None]


        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_template[None,None])   # (B,T,L,L,3)
        t2d = torch.cat((t2d, torch.zeros((1,1,L,L,3))), dim=-1) #three extra dimensions: adjacent, not adjacent, masked
        t2d[...,-1] = 1 # set whole block adjacency to mask token
        # t2d[...,0] = 1  # for the buggy model, set this feature to zero instead of 2 


        ### idx ###
        ###########
        idx = torch.arange(L)[None]
        # index jumps for chains 
        cur_sum=0
        for Lchain in self.chain_lengths:
            idx[:,cur_sum+Lchain:] += 200
            cur_sum += Lchain

        ### alpha_t ###
        ###############
        # get the torsions 
        xyz_template =xyz_template[None, None]
        xyz_template = torch.cat((xyz_template, torch.full((1,1,L,13,3), float('nan'))), dim=3)
        

        alpha, _, alpha_mask, _ = util.get_torsions(xyz_template.reshape(-1,L,27,3), self.query[None], TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
        
        # device placement 
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        xyz_template = xyz_template.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        
        return msa_masked, msa_full, self.query[None].clone().to(self.device), torch.squeeze(xyz_template, dim=0), idx, t1d, t2d, xyz_template, alpha_t

class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB
    """

    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''

        run_rf2 = self.should_we_run_rf2( t, self.diffuser.T, self.diffuser_conf.partial_T )

        if run_rf2:

            msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
                seq_t, x_t, t)
            B,N,L = xyz_t.shape[:3]

            ##################################
            ######## Str Self Cond ###########
            ##################################
            if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):
                #ic('Providing Self Cond')
                    
                zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                xyz_t = torch.cat((self.prev_pred.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

                t2d_44   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
            else:
                xyz_t = torch.zeros_like(xyz_t)
                t2d_44   = torch.zeros_like(t2d[...,:44])
            # No effect if t2d is only dim 44
            t2d[...,:44] = t2d_44

            ##################################
            ######## Seq Self Cond ###########
            ##################################
            if self.seq_self_cond:
                if t < self.diffuser.T:
                    ic('Providing Seq Self Cond')
            
                    t1d[:,:,:,:20] = self.prev_seq_pred # [B,T,L,d_t1d]
                    t1d[:,:,:,20]  = 0 # Setting mask token to zero
            
                else:
                    t1d[:,:,:,:21] = 0

            if self.symmetry is not None:
                idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

            with torch.no_grad():
                px0=xt_in
                for rec in range(self.recycle_schedule[t-1]):
                    msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                        msa_full,
                                        seq_in,
                                        px0,
                                        idx_pdb,
                                        t1d=t1d,
                                        t2d=t2d,
                                        xyz_t=xyz_t,
                                        alpha_t=alpha_t,
                                        msa_prev = None,
                                        pair_prev = None,
                                        state_prev = None,
                                        t=torch.tensor(t),
                                        return_infer=True,
                                        motif_mask=self.diffusion_mask.squeeze().to(self.device))   

                    if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
                        px0 = self.symmetrise_prev_pred(px0=px0,seq_in=seq_in, alpha=alpha)[:,:,:3]

                    # To permit 'recycling' within a timestep, in a manner akin to how this model was trained
                    # Aim is to basically just replace the xyz_t with the model's last px0, and to *not* recycle the state, pair or msa embeddings
                    if rec < self.recycle_schedule[t-1] -1:
                        zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                        xyz_t = torch.cat((px0.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

                        t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
                        if self.seq_self_cond:
                            # Allow this model to also do sequence recycling

                            t1d[:,:,:,:20] = logits[:,None,:,:20]
                            t1d[:,:,:,20]  = 0 # Setting mask token to zero

                        px0=xt_in

            self.prev_pred = torch.clone(px0)

            self.last_rf2_vars = (msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt, seq_in)
        else:
            assert hasattr( self, "last_rf2_vars" )
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt, seq_in = self.last_rf2_vars


        # prediction of X0
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]

        if self.seq_diffuser is None:
            # Default method of decoding sequence
            seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
            sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

            pseq_0 = torch.nn.functional.one_hot(
                sampled_seq, num_classes=22).to(self.device).float()

            pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze()].to(self.device) # [L,22]
        else:
            # Sequence Diffusion
            pseq_0 = logits.squeeze()
            pseq_0 = pseq_0[:,:20]

            pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze(),:20].to(self.device)

            sampled_seq = torch.argmax(pseq_0, dim=-1)
        
        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_diffusion_mask=self.mask_seq.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
            self._log.info(
                    f'Timestep {t}, sequence input to next step:  { seq2chars(torch.argmax(seq_t_1, dim=-1).tolist())}')
            self._log.info(
                    f'Timestep {t}, structure input to next step: {"".join(["+" if i else "-" for i in self.mask_str.squeeze()])}')
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)

        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        return px0, x_t_1, seq_t_1, tors_t_1, plddt

class JWStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by JW in
    frame_sql2_pdb_data_T200_sinusoidal_frozenmotif_lddt_distog_noseq_physics_selfcond_lowlr_train_session2022-10-06_1665075957.8513756
    """
    def __init__(self, conf: DictConfig):
        super().__init__(conf)
        self.self_cond = self.inf_conf.use_jw_selfcond 
        if not self.self_cond:
            print(" ", "*" * 100, " ", "WARNING: You're using the JWStyleSelfCond sampler, but inference.use_jw_selfcond is set to False. Is this intentional?", " ", "*" * 100, " ", sep=os.linesep)
    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_t (torch.tensor): (L) The initialized sequence used in updating the sequence.
            
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        out = self._preprocess(seq_t, x_t, t)
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)
        # Save inputs for next timestep
        self.t1d = t1d[:,:1]
        self.t2d = t2d[:,:1]
        self.alpha = alpha_t
        N,L = msa_masked.shape[:2]

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)
        with torch.no_grad():
            px0=xt_in
            for _ in range(self.recycle_schedule[t-1]):
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = self.msa_prev,
                                    pair_prev = self.pair_prev,
                                    state_prev = self.state_prev,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))

        self.msa_prev=msa_prev
        self.pair_prev=pair_prev
        self.state_prev=state_prev
        self.prev_pred = torch.clone(px0)
        # prediction of X0 
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]
        #sampled_seq = torch.argmax(logits.squeeze(), dim=-1)
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position 

        # grab only the query sequence prediction - adjustment for Seq2StrSampler
        sampled_seq = sampled_seq.reshape(N,L,-1)[0,0]

        # Process outputs.
        mask_seq = self.mask_seq
        sampled_seq[mask_seq.squeeze()] = seq_init[
            mask_seq.squeeze()].to(self.device)

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device)

        seq_t = torch.nn.functional.one_hot(
            seq_t, num_classes=22).to(self.device)

        self._log.info(
            f'Timestep {t}, current sequence: { seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')

        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = torch.clone(sampled_seq)
            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)
        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
        return px0, x_t_1, seq_t_1, tors_t_1, plddt

    def _preprocess(self, seq, xyz_t, t):
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = super()._preprocess(seq, xyz_t, t)
        t1d = torch.cat((t1d, torch.zeros_like(t1d[...,:1])), dim=-1)
        if t != self.T and self.self_cond:
            # add last step
            xyz_prev_padded = torch.full_like(xyz_t, float('nan'))
            xyz_prev_padded[:,:,:,:3,:] = self.prev_pred[None] 
            xyz_t = torch.cat((xyz_t, xyz_prev_padded), dim=1)
            t1d = t1d.repeat(1,2,1,1)
            t1d[:,1,:,21] = self.t1d[...,21]
            t1d[:,1,:,22] = 1
            t2d_temp = xyz_to_t2d(xyz_prev_padded).to(self.device, non_blocking=True)
            t2d = torch.cat((t2d, t2d_temp), dim=1)
            alpha_temp = torch.zeros_like(alpha_t).to(self.device, non_blocking=True)
            alpha_t = torch.cat((alpha_t, alpha_temp), dim=1)
        return msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t

class ScaffoldedSampler(NRBStyleSelfCond):
    """ 
    Model Runner for Scaffold-Constrained diffusion
    """
    def __init__(self, conf: DictConfig):
        """
        Initialize scaffolded sampler, which inherits from Sampler
        """
        super().__init__(conf)
        
        # Now have the option to provide residue-specific secondary structure
        
        if conf.scaffoldguided.scaffold_dir is None:
            assert any(x is not None for x in (conf.contigmap.inpaint_str_helix, conf.contigmap.inpaint_str_strand, conf.contigmap.inpaint_str_loop))
            if conf.contigmap.inpaint_str_loop is not None:
                assert conf.scaffoldguided.mask_loops == False, "You shouldn't be masking loops if you're specifying loop secondary structure"
        else:
            # initialize BlockAdjacency sampling class
            assert all(x is None for x in (conf.contigmap.inpaint_str_helix, conf.contigmap.inpaint_str_strand, conf.contigmap.inpaint_str_loop)), "can't provide scaffold_dir if you're also specifying per-residue ss"
            self.blockadjacency = iu.BlockAdjacency(conf.scaffoldguided, conf.inference.num_designs)

        if conf.scaffoldguided.target_pdb:
            self.target = iu.Target(conf.scaffoldguided, conf.ppi.hotspot_res)
            self.target_pdb = self.target.get_target()
            if conf.scaffoldguided.target_ss is not None:
                self.target_ss = torch.load(conf.scaffoldguided.target_ss).long()
                self.target_ss = torch.nn.functional.one_hot(self.target_ss, num_classes=4)
            if conf.scaffoldguided.target_adj is not None:
                self.target_adj = torch.load(conf.scaffoldguided.target_adj).long()
                self.target_adj=torch.nn.functional.one_hot(self.target_adj, num_classes=3)
        else:
            self.target = None
            self.target_pdb=False

    def sample_init(self):
        """
        Wrapper method for taking ss + adj, and outputting xt, seq_t
        """
        if hasattr(self, 'blockadjacency'):
            self.L, self.ss, self.adj = self.blockadjacency.get_scaffold()
            self.adj = nn.one_hot(self.adj.long(), num_classes=3)
        else:
            self.L=100 # shim. Get's overwritten
        xT = torch.full((self.L, 27,3), np.nan)
        xT = get_init_xyz(xT[None,None]).squeeze()
        seq_T = torch.full((self.L,),21)
        self.diffusion_mask = torch.full((self.L,),False)
        atom_mask = torch.full((self.L,27), False)
        self.binderlen=self.L
        # for ppi
        self.binder_L = np.copy(self.L)
        if self.target:
            target_L = np.shape(self.target_pdb['xyz'])[0]
            target_xyz = torch.full((target_L, 27, 3), np.nan)
            target_xyz[:,:14,:] = torch.from_numpy(self.target_pdb['xyz'])
            xT = torch.cat((xT, target_xyz), dim=0)
            seq_T = torch.cat((seq_T, torch.from_numpy(self.target_pdb['seq'])), dim=0)
            self.diffusion_mask = torch.cat((self.diffusion_mask, torch.full((target_L,), True)),dim=0)
            mask_27 = torch.full((target_L, 27), False)
            mask_27[:,:14] = torch.from_numpy(self.target_pdb['mask'])
            atom_mask = torch.cat((atom_mask, mask_27), dim=0)
            self.L += target_L
        if self.contig_conf.contigs is None and self.contig_conf.specific_contig_pkl is None: 
            # make contigmap object
            if self.target:
                contig = []
                for idx,i in enumerate(self.target_pdb['pdb_idx'][:-1]):
                    if idx==0:
                        start=i[1]               
                    if i[1] + 1 != self.target_pdb['pdb_idx'][idx+1][1] or i[0] != self.target_pdb['pdb_idx'][idx+1][0]:
                        contig.append(f'{i[0]}{start}-{i[1]},0 ')
                        start = self.target_pdb['pdb_idx'][i+1][1]
                contig.append(f"{self.target_pdb['pdb_idx'][-1][0]}{start}-{self.target_pdb['pdb_idx'][-1][1]},0 ")
                contig.append(f"{self.binderlen}-{self.binderlen}")
                contig = ["".join(contig)]
            else:
                contig = [f"{self.binderlen}-{self.binderlen}"]
            self.contig_map=ContigMap(self.target_pdb, contig)
            self.mappings = self.contig_map.get_mappings()
            L_mapped=len(self.contig_map.ref)
            self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None,:]
            self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None,:]
            self.binderlen =  len(self.contig_map.inpaint)
            self.L = len(self.contig_map.inpaint_seq)

        else:
            # get contigmap from command line
            assert self.target is None, "Giving a target is the wrong way of handling this is you're doing contigs and secondary structure"

            # process target and reinitialise potential_manager. This is here because the 'target' is always set up to be the second chain in out inputs. Could change this down the line
            self.target_feats = iu.process_target(self.inf_conf.input_pdb)
            # moved this here as should be updated each iteration of diffusion
            self.contig_map = self.construct_contig(self.target_feats)
            self.mappings = self.contig_map.get_mappings()
            self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None,:]
            self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None,:]
            self.binderlen =  len(self.contig_map.inpaint)
            self.L = len(self.contig_map.inpaint_seq) 
            target_feats = self.target_feats
            contig_map = self.contig_map

            xyz_27 = target_feats['xyz_27']
            mask_27 = target_feats['mask_27']
            seq_orig = target_feats['seq']
            L_mapped = len(self.contig_map.ref)
            seq_T=torch.full((L_mapped,),21)
            seq_T[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
            seq_T[~self.mask_seq.squeeze()] = 21
            diffusion_mask = self.mask_str
            self.diffusion_mask = diffusion_mask
            
            xT = torch.full((1,1,L_mapped,27,3), np.nan)
            xT[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0,...]
            xT = get_init_xyz(xT).squeeze()
            # adjust the size of the input atom map
            atom_mask = torch.full((L_mapped, 27), False)
            atom_mask[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

            if hasattr(self.contig_map, 'ss_spec'):
                self.adj=torch.full((L_mapped, L_mapped),2) # masked
                self.adj=nn.one_hot(self.adj.long(), num_classes=3)
                self.ss=iu.ss_from_contig(self.contig_map.ss_spec)
            assert L_mapped==self.adj.shape[0]
 
        self.hotspot_0idx=iu.get_idx0_hotspots(self.mappings, self.ppi_conf, self.binderlen)
        # Now initialise potential manager here. This allows variable-length binder design
        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.hotspot_0idx,
                                                  self.binderlen)
        if not self.contig_map.specific_mappings:      
            self.chain_idx=['A' if i < self.binderlen else 'B' for i in range(self.L)]
        else:
            self.chain_idx=[i[0] for i in self.contig_map.hal]
            self.out_idx=[int(i[1]) for i in self.contig_map.hal]
        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input+1)
        if self.seq_diffuser is None:
            seq_T=torch.nn.functional.one_hot(seq_T, num_classes=22).float()
        else:
            raise NotImplementedError("seq diffusion and scaffold-conditioned generation not implemented")

        fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
            xT,
            torch.clone(seq_T),  # TODO: Check if copy is needed.
            atom_mask.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)


        # Setup denoiser 
        is_motif = self.mask_seq.squeeze()
        is_shown_at_t = torch.tensor(aa_masks[-1])
        visible = is_motif | is_shown_at_t
        if self.diffuser_conf.partial_T:
            seq_t[visible] = seq_orig[visible]

        self.denoiser = self.construct_denoiser(self.L, visible=visible)


        xT = torch.clone(fa_stack[-1].squeeze()[:,:14,:])
        return xT, seq_T
    
    def _preprocess(self, seq, xyz_t, t):
        msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t = super()._preprocess(seq, xyz_t, t, repack=False)
        
        # Now just need to tack on ss/adj
        assert self.preprocess_conf.d_t1d == 28, "The checkpoint you're using hasn't been trained with SS/block adjacency features"
        assert self.preprocess_conf.d_t2d == 47, "The checkpoint you're using hasn't been trained with SS/block adjacency features"
        
        if self.target:
            blank_ss = torch.nn.functional.one_hot(torch.full((self.L-self.binderlen,), 3), num_classes=4)
            full_ss = torch.cat((self.ss, blank_ss), dim=0)
            if self._conf.scaffoldguided.target_ss is not None:
                full_ss[self.binderlen:] = self.target_ss
        else:
            full_ss = self.ss
        t1d=torch.cat((t1d, full_ss[None,None].to(self.device)), dim=-1)
        t1d = t1d.float()
        ### t2d ###
        ###########

        if self.d_t2d == 47:
            if self.target:
                full_adj = torch.zeros((self.L, self.L, 3))
                full_adj[:,:,-1] = 1. #set to mask
                full_adj[:self.binderlen, :self.binderlen] = self.adj
                if self._conf.scaffoldguided.target_adj is not None:
                    full_adj[self.binderlen:,self.binderlen:] = self.target_adj
            else:
                full_adj = self.adj
            t2d=torch.cat((t2d, full_adj[None,None].to(self.device)),dim=-1)

        ### idx ###
        ###########
        if self.target:
            idx_pdb[:,self.binderlen:] += 200

        ### msa N/C ###
        ###############
        msa_masked[...,-2:] = 0
        msa_masked[...,0,-2] = 1 # N ter token
        msa_masked[...,self.binderlen-1,-1] = 1 # C ter token

        msa_full[...,-2:] = 0
        msa_full[...,0,-2] = 1 # N ter token
        msa_full[...,self.binderlen-1,-1] = 1 # C ter token

        return msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t
