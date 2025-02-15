import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel


CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

"""# **Model**"""

class labse_clip(torch.nn.Module):
    def __init__(self, hdim, args):
        super().__init__()
        self.labse = SentenceTransformer(LABSE_MODEL_NAME)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        self.freeze_encoders()
        self.args = args

        # learnt parameters
        self.mlps = torch.nn.ModuleList([
            torch.nn.Linear(hdim, hdim),
            torch.nn.Linear(hdim, hdim),
            torch.nn.Linear(hdim, hdim),
            torch.nn.Linear(hdim, hdim),
            torch.nn.Linear(hdim, hdim),
            torch.nn.Linear(hdim, hdim),
        ])

    def freeze_encoders(self):
        for param in self.labse.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):

        org_eng_labse_emb = inputs['eng_labse_emb']
        new_eng_labse_emb = org_eng_labse_emb

        # normalize clip text emb
        eng_clip_emb = inputs['eng_clip_emb']
        eng_clip_emb = torch.nn.functional.normalize(eng_clip_emb, dim=-1, p=2)

        new_eng_labse_emb = self.inference(new_eng_labse_emb)
        
        return {
            'non_eng_labse_emb': inputs['non_eng_labse_emb'],
            'new_eng_labse_emb': new_eng_labse_emb,
            'org_eng_labse_emb': org_eng_labse_emb,
            'eng_clip_emb': eng_clip_emb,
        }

    
    def inference(self, labse_emb):

        for idx, mlp in enumerate(self.mlps):
            if self.args.emb_method == "hybrid":
                # first and last 2 layers
                # do only linear
                if idx < 2:
                    labse_emb = mlp(labse_emb)
                else:
                    # skip conn
                    labse_emb = labse_emb + mlp(labse_emb)
            else:
                raise ValueError("Only Hybrid method is not supported in model_hybrid.py")

        labse_emb = torch.nn.functional.normalize(labse_emb, dim=-1, p=2)

        return labse_emb


    def mse_loss(self, outputs):
        return torch.nn.functional.mse_loss(outputs['new_eng_labse_emb'], outputs['eng_clip_emb'])

    def kl_loss(self, outputs, use_mse=False):
        # diff batch size is allowed

        # B x D
        new_eng_labse_emb = outputs['new_eng_labse_emb']
        org_eng_labse_emb = outputs['org_eng_labse_emb']

        # B' x D
        non_eng_labse_emb = outputs['non_eng_labse_emb']

        if len(new_eng_labse_emb.shape) == 3 and new_eng_labse_emb.shape[1] == 1:
            new_eng_labse_emb = new_eng_labse_emb.squeeze(1)

        if len(org_eng_labse_emb.shape) == 3 and org_eng_labse_emb.shape[1] == 1:
            org_eng_labse_emb = org_eng_labse_emb.squeeze(1)

        if len(non_eng_labse_emb.shape) == 3 and non_eng_labse_emb.shape[1] == 1:
            non_eng_labse_emb = non_eng_labse_emb.squeeze(1)

        # B x B'
        # predicted distribution
        eps = 1e-8
        pred_sim_score = new_eng_labse_emb @ non_eng_labse_emb.permute(1, 0)
        # pred_sim_score = pred_sim_score.exp().softmax(-1)

        # B x B'
        # true distribution
        gt_sim_score = org_eng_labse_emb @ non_eng_labse_emb.permute(1, 0)
        # gt_sim_score = gt_sim_score.exp().softmax(-1)


        # batchmean reduction in pytorch implementation
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        if use_mse:
            kl_loss = torch.nn.functional.mse_loss(pred_sim_score, gt_sim_score)
        else:
            kl_loss = (gt_sim_score * (gt_sim_score.log() - pred_sim_score.log())).sum(-1).mean()

        # print("pred_sim_score")
        # print(pred_sim_score)
        # print("-"*100)
        # print("gt_sim_score")
        # print(gt_sim_score)
        # print("-"*100)
        # print("gt_sim_score.log() - pred_sim_score.log()")
        # print(gt_sim_score.log() - pred_sim_score.log())
        # print(torch.isnan(gt_sim_score.log() - pred_sim_score.log()).sum())
        # print("-"*100)
        # print("gt_sim_score * (gt_sim_score.log() - pred_sim_score.log())")
        # print(torch.isnan(gt_sim_score * (gt_sim_score.log() - pred_sim_score.log())).sum())

        # print("="*100)

        return kl_loss

