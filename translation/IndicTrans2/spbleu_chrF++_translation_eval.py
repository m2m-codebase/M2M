
from sacrebleu import corpus_bleu, corpus_chrf
import sys
import pandas as pd


csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# (ref col, hyp col) pairs

# IndicTrans2
col_pairs = [
    ['caption', 'ben_Beng_to_eng_Latn'],
    ['caption', 'guj_Gujr_to_eng_Latn'],
    ['caption', 'hin_Deva_to_eng_Latn'],
    ['caption', 'kan_Knda_to_eng_Latn'],
    ['caption', 'mal_Mlym_to_eng_Latn'],
    ['caption', 'mar_Deva_to_eng_Latn'],
    ['caption', 'npi_Deva_to_eng_Latn'],
    ['caption', 'pan_Guru_to_eng_Latn'],
    ['caption', 'tam_Taml_to_eng_Latn'],
    ['caption', 'tel_Telu_to_eng_Latn'],
    ['caption', 'urd_Arab_to_eng_Latn'],
 ]
'''
# AYA
col_pairs = [
    ['eng_Latn-arb_Arab-sentence_arb_Arab', 'aya_translated_eng_Latn-arb_Arab-sentence_arb_Arab'],
    ['eng_Latn-zho_Hans-sentence_zho_Hans', 'aya_translated_eng_Latn-zho_Hans-sentence_zho_Hans'],
    ['eng_Latn-zho_Hant-sentence_zho_Hant', 'aya_translated_eng_Latn-zho_Hant-sentence_zho_Hant'],
    ['eng_Latn-ces_Latn-sentence_ces_Latn', 'aya_translated_eng_Latn-ces_Latn-sentence_ces_Latn'],
    ['eng_Latn-nld_Latn-sentence_nld_Latn', 'aya_translated_eng_Latn-nld_Latn-sentence_nld_Latn'],
    ['eng_Latn-fra_Latn-sentence_fra_Latn', 'aya_translated_eng_Latn-fra_Latn-sentence_fra_Latn'],
    ['eng_Latn-deu_Latn-sentence_deu_Latn', 'aya_translated_eng_Latn-deu_Latn-sentence_deu_Latn'],
    ['eng_Latn-ell_Grek-sentence_ell_Grek', 'aya_translated_eng_Latn-ell_Grek-sentence_ell_Grek'],
    ['eng_Latn-heb_Hebr-sentence_heb_Hebr', 'aya_translated_eng_Latn-heb_Hebr-sentence_heb_Hebr'],
    ['eng_Latn-hin_Deva-sentence_hin_Deva', 'aya_translated_eng_Latn-hin_Deva-sentence_hin_Deva'],
    ['eng_Latn-ind_Latn-sentence_ind_Latn', 'aya_translated_eng_Latn-ind_Latn-sentence_ind_Latn'],
    ['eng_Latn-ita_Latn-sentence_ita_Latn', 'aya_translated_eng_Latn-ita_Latn-sentence_ita_Latn'],
    ['eng_Latn-jpn_Jpan-sentence_jpn_Jpan', 'aya_translated_eng_Latn-jpn_Jpan-sentence_jpn_Jpan'],
    ['eng_Latn-kor_Hang-sentence_kor_Hang', 'aya_translated_eng_Latn-kor_Hang-sentence_kor_Hang'],
    ['eng_Latn-pes_Arab-sentence_pes_Arab', 'aya_translated_eng_Latn-pes_Arab-sentence_pes_Arab'],
    ['eng_Latn-pol_Latn-sentence_pol_Latn', 'aya_translated_eng_Latn-pol_Latn-sentence_pol_Latn'],
    ['eng_Latn-por_Latn-sentence_por_Latn', 'aya_translated_eng_Latn-por_Latn-sentence_por_Latn'],
    ['eng_Latn-ron_Latn-sentence_ron_Latn', 'aya_translated_eng_Latn-ron_Latn-sentence_ron_Latn'],
    ['eng_Latn-rus_Cyrl-sentence_rus_Cyrl', 'aya_translated_eng_Latn-rus_Cyrl-sentence_rus_Cyrl'],
    ['eng_Latn-spa_Latn-sentence_spa_Latn', 'aya_translated_eng_Latn-spa_Latn-sentence_spa_Latn'],
    ['eng_Latn-tur_Latn-sentence_tur_Latn', 'aya_translated_eng_Latn-tur_Latn-sentence_tur_Latn'],
    ['eng_Latn-ukr_Cyrl-sentence_ukr_Cyrl', 'aya_translated_eng_Latn-ukr_Cyrl-sentence_ukr_Cyrl'],
    ['eng_Latn-vie_Latn-sentence_vie_Latn', 'aya_translated_eng_Latn-vie_Latn-sentence_vie_Latn'],
]
 
'''
spbleu_list = []
chrf2_list = []
for ref_col, hyp_col in col_pairs:
    print("Processing", ref_col, hyp_col)

    references = df[ref_col].tolist()
    references = [[x] for x in references]

    hypotheses = df[hyp_col].tolist()
    hypotheses = [str(x) for x in hypotheses]
    print(hypotheses[:2])
    print("-"*100)
    print(references[:2])

    spbleu_score = corpus_bleu(hypotheses, references, tokenize="spm")  # "spm" for sentence-piece tokenization

    chrf2_score = corpus_chrf(hypotheses, references, beta=2)  # beta=2 balances precision and recall

    print(f"spBLEU score: {spbleu_score.score}")
    print(f"chrF++ score: {chrf2_score.score}")

    spbleu_list.append(spbleu_score.score)
    chrf2_list.append(chrf2_score.score)
    print("="*100)


print("Mean spBLEU", sum(spbleu_list)/len(spbleu_list))
print("Mean chrF++", sum(chrf2_list)/len(chrf2_list))



langs = "*".join([x.split('_')[0] for _, x in col_pairs])
print(langs)
print("*".join([f"{x:.1f}" for x in spbleu_list]))
print("*".join([f"{x:.1f}" for x in chrf2_list]))
