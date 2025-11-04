#%%
_symbols = ['_', 'a', 'aj', 'aw', 'aː', 
            'b', 'bʲ', 'c', 'cʰ', 
            'cʷ', 'd', 'dʒ', 'dʲ', 
            'd̪', 'e', 'ej', 'eː', 
            'f', 'fʲ', 'h', 'i', 
            'iː', 'j', 'k', 'kp', 
            'kʰ', 'kʷ', 'l', 'm', 
            'mʲ', 'm̩', 'n', 'o', 
            'ow', 'oː', 'p', 'pʰ', 
            'pʲ', 'pʷ', 's', 't', 
            'tʃ', 'tʰ', 'tʲ', 'tʷ', 
            't̪', 'u', 'uː', 'v', 
            'vʲ', 'w', 'z', 'æ', 
            'ç', 'ð', 'ŋ', 'ɐ', 
            'ɑ', 'ɑː', 'ɒ', 'ɒː', 
            'ɔ', 'ɔj', 'ɖ', 'ə', 
            'əw', 'ɚ', 'ɛ', 'ɛː', 
            'ɜ', 'ɜː', 'ɝ', 'ɟ', 
            'ɟʷ', 'ɡ', 'ɡʷ', 'ɪ', 
            'ɫ', 'ɲ', 'ɹ', 'ɾ', 
            'ʃ', 'ʈ', 'ʈʲ', 'ʈʷ', 
            'ʉ', 'ʉː', 'ʊ', 'ʋ', 
            'ʎ', 'ʒ', 'θ', ]
start_symbols = [
        'S_a', 'S_aj', 'S_aw', 
        'S_aː', 'S_b', 'S_bʲ', 'S_c', 
        'S_cʰ', 'S_cʷ', 'S_d', 'S_dʒ', 
        'S_dʲ', 'S_d̪', 'S_e', 'S_ej', 
        'S_eː', 'S_f', 'S_fʲ', 'S_h', 
        'S_i', 'S_iː', 'S_j', 'S_k', 
        'S_kp', 'S_kʰ', 'S_kʷ', 'S_l', 
        'S_m', 'S_mʲ', 'S_m̩', 'S_n', 
        'S_o', 'S_ow', 'S_oː', 'S_p', 
        'S_pʰ', 'S_pʲ', 'S_pʷ', 'S_s', 
        'S_t', 'S_tʃ', 'S_tʰ', 'S_tʲ', 
        'S_tʷ', 'S_t̪', 'S_u', 'S_uː', 
        'S_v', 'S_vʲ', 'S_w', 'S_z', 
        'S_æ', 'S_ç', 'S_ð', 'S_ŋ', 
        'S_ɐ', 'S_ɑ', 'S_ɑː', 'S_ɒ', 
        'S_ɒː', 'S_ɔ', 'S_ɔj', 'S_ɖ', 
        'S_ə', 'S_əw', 'S_ɚ', 'S_ɛ', 
        'S_ɛː', 'S_ɜ', 'S_ɜː', 'S_ɝ', 
        'S_ɟ', 'S_ɟʷ', 'S_ɡ', 'S_ɡʷ', 
        'S_ɪ', 'S_ɫ', 'S_ɲ', 'S_ɹ', 
        'S_ɾ', 'S_ʃ', 'S_ʈ', 'S_ʈʲ', 
        'S_ʈʷ', 'S_ʉ', 'S_ʉː', 'S_ʊ', 
        'S_ʋ', 'S_ʎ', 'S_ʒ', 'S_θ', 'S_spn'
        ]
mid_symbols = [
    'M_a', 'M_aj', 'M_aw', 
    'M_aː', 'M_b', 'M_bʲ', 'M_c', 
    'M_cʰ', 'M_cʷ', 'M_d', 'M_dʒ', 
    'M_dʲ', 'M_d̪', 'M_e', 'M_ej', 
    'M_eː', 'M_f', 'M_fʲ', 'M_h', 
    'M_i', 'M_iː', 'M_j', 'M_k', 
    'M_kp', 'M_kʰ', 'M_kʷ', 'M_l', 
    'M_m', 'M_mʲ', 'M_m̩', 'M_n', 
    'M_o', 'M_ow', 'M_oː', 'M_p', 
    'M_pʰ', 'M_pʲ', 'M_pʷ', 'M_s', 
    'M_t', 'M_tʃ', 'M_tʰ', 'M_tʲ', 
    'M_tʷ', 'M_t̪', 'M_u', 'M_uː', 
    'M_v', 'M_vʲ', 'M_w', 'M_z', 
    'M_æ', 'M_ç', 'M_ð', 'M_ŋ', 
    'M_ɐ', 'M_ɑ', 'M_ɑː', 'M_ɒ', 
    'M_ɒː', 'M_ɔ', 'M_ɔj', 'M_ɖ', 
    'M_ə', 'M_əw', 'M_ɚ', 'M_ɛ', 
    'M_ɛː', 'M_ɜ', 'M_ɜː', 'M_ɝ', 
    'M_ɟ', 'M_ɟʷ', 'M_ɡ', 'M_ɡʷ', 
    'M_ɪ', 'M_ɫ', 'M_ɲ', 'M_ɹ', 
    'M_ɾ', 'M_ʃ', 'M_ʈ', 'M_ʈʲ', 
    'M_ʈʷ', 'M_ʉ', 'M_ʉː', 'M_ʊ', 
    'M_ʋ', 'M_ʎ', 'M_ʒ', 'M_θ', "M_spn"
]
end_symbols = [
    'E_a', 'E_aj', 'E_aw', 
    'E_aː', 'E_b', 'E_bʲ', 'E_c', 
    'E_cʰ', 'E_cʷ', 'E_d', 'E_dʒ', 
    'E_dʲ', 'E_d̪', 'E_e', 'E_ej', 
    'E_eː', 'E_f', 'E_fʲ', 'E_h', 
    'E_i', 'E_iː', 'E_j', 'E_k', 
    'E_kp', 'E_kʰ', 'E_kʷ', 'E_l', 
    'E_m', 'E_mʲ', 'E_m̩', 'E_n', 
    'E_o', 'E_ow', 'E_oː', 'E_p', 
    'E_pʰ', 'E_pʲ', 'E_pʷ', 'E_s', 
    'E_t', 'E_tʃ', 'E_tʰ', 'E_tʲ', 
    'E_tʷ', 'E_t̪', 'E_u', 'E_uː', 
    'E_v', 'E_vʲ', 'E_w', 'E_z', 
    'E_æ', 'E_ç', 'E_ð', 'E_ŋ', 
    'E_ɐ', 'E_ɑ', 'E_ɑː', 'E_ɒ', 
    'E_ɒː', 'E_ɔ', 'E_ɔj', 'E_ɖ', 
    'E_ə', 'E_əw', 'E_ɚ', 'E_ɛ', 
    'E_ɛː', 'E_ɜ', 'E_ɜː', 'E_ɝ', 
    'E_ɟ', 'E_ɟʷ', 'E_ɡ', 'E_ɡʷ', 
    'E_ɪ', 'E_ɫ', 'E_ɲ', 'E_ɹ', 
    'E_ɾ', 'E_ʃ', 'E_ʈ', 'E_ʈʲ', 
    'E_ʈʷ', 'E_ʉ', 'E_ʉː', 'E_ʊ', 
    'E_ʋ', 'E_ʎ', 'E_ʒ', 'E_θ', 'E_spn'
]
space = ['_']
symbols = start_symbols + mid_symbols + end_symbols + space
_symbol_to_id = {s : i for i, s in enumerate(symbols)}
_id_to_symbol = {i : s for i, s in enumerate(symbols)}


def phoneme_to_sequence(phonemes):
    return [_symbol_to_id[p] for p in phonemes]

def sequence_to_phoneme(sequence):
    return [_id_to_symbol[s] for s in sequence]

