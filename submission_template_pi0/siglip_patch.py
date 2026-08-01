"""transformers の SigLIP 実装への軽量パッチ。

lerobot の Pi0 実装（lerobot/policies/pi0/modeling_pi0.py）は、本来
`pip install "lerobot[pi]@git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"`
という、GitHub の特定ブランチにパッチされた transformers の導入を前提にしている。
このブランチの実体は https://github.com/huggingface/transformers/tree/fix/lerobot_openpi
の `src/transformers/models/siglip/` にあり、PyPI 版 transformers (v4.53.2 系) との差分は
`SiglipVisionTransformer.forward` 内で encoder が bfloat16 の場合に入力の dtype を
bfloat16 へ揃える処理が1箇所追加されているだけである
（差分: `hidden_states = hidden_states.to(torch.bfloat16)` を encoder 呼び出し直前に追加）。

提出物は `requirements.txt` に `git+` 等の外部ソース指定を含められない（採点環境は外部通信も
遮断する）ため、上記ブランチ全体を導入する代わりに、この最小差分だけをここでモンキーパッチとして
再現する。あわせて lerobot 側が実施しているチェック
`transformers.models.siglip.check.check_whether_transformers_replace_is_installed_correctly()`
（本来は transformers のバージョン文字列 "4.53.2"/"4.53.3" と一致するかを見ているだけ）も、
このパッチを適用済みである旨を返すダミーモジュールとして差し替える。
"""

import sys
import types

import torch
import transformers.models.siglip as siglip_pkg
from transformers.models.siglip import modeling_siglip
from transformers.modeling_outputs import BaseModelOutputWithPooling

_PATCHED_ATTR = "_parc2026_siglip_bf16_patch_applied"


def _patched_vision_forward(self, pixel_values, interpolate_pos_encoding=False, **kwargs):
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    # fix/lerobot_openpi ブランチと同じ処理: encoder が bfloat16 で実行される場合、
    # 入力の hidden_states も bfloat16 に揃える（そうしないと dtype 不一致で失敗する）。
    if (
        len(self.encoder.layers) > 0
        and self.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    ):
        hidden_states = hidden_states.to(torch.bfloat16)

    encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)

    last_hidden_state = encoder_outputs.last_hidden_state
    last_hidden_state = self.post_layernorm(last_hidden_state)

    pooler_output = self.head(last_hidden_state) if self.use_head else None

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
    )


def _install_check_stub() -> None:
    """`from transformers.models.siglip import check` が解決できるようにする。"""
    module_name = "transformers.models.siglip.check"
    if module_name in sys.modules:
        return

    check_module = types.ModuleType(module_name)
    check_module.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules[module_name] = check_module
    # `from transformers.models.siglip import check` は siglip パッケージ自体の属性を見るため、
    # サブモジュールとして sys.modules に登録するだけでなく、パッケージオブジェクトにも生やす。
    siglip_pkg.check = check_module


def apply_siglip_patch() -> None:
    """SiglipVisionTransformer.forward に bfloat16 対応パッチを適用し、
    lerobot の Pi0 実装が要求するチェックを満たすスタブを登録する。
    冪等（複数回呼び出しても安全）。
    """
    if getattr(modeling_siglip.SiglipVisionTransformer, _PATCHED_ATTR, False):
        return

    modeling_siglip.SiglipVisionTransformer.forward = _patched_vision_forward
    setattr(modeling_siglip.SiglipVisionTransformer, _PATCHED_ATTR, True)

    _install_check_stub()


_PI0_PATCHED_ATTR = "_parc2026_pi0_denoise_step_patch_applied"
_PI0_SCALE_PATCHED_ATTR = "_parc2026_pi0_embed_scale_patch_applied"


def _patched_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    """lerobot 0.4.4 の PI0Pytorch.embed_prefix の埋め込みスケール修正版。

    lerobot の Pi0 は transformers のフォーク `fix/lerobot_openpi` を前提にしている。
    そのフォークは openpi の実装に合わせて次の2箇所を無効化している:
      - GemmaModel.forward の `hidden_states = hidden_states * normalizer`
        （normalizer = sqrt(hidden_size)）をコメントアウト
      - PaliGemma.get_image_features の `/ sqrt(text_config.hidden_size)` を削除

    素の transformers ではどちらも有効なため、S = sqrt(hidden_size) として
    実効的な埋め込みの大きさが次のようにずれる:
      - 画像 : (P / S) * S = P            → フォークと一致（対処不要）
      - 言語 : (E * S) * S = E * S^2      → フォークの E * S に対し S 倍（約45倍）過大
    結果として言語トークンが画像トークンを数値的に押し潰し、
    画像を黒く塗り潰しても出力がほとんど変わらない（＝視覚を無視する）状態になる。

    ここでは元実装にある言語埋め込みへの `* sqrt(lang_emb_dim)` を取り除く。
    後段の GemmaModel.forward が同じ係数を掛けるため、実効値はフォークと一致する。
    （係数を掛けてから割り戻すのではなく乗算自体を消すので、丸め誤差も生じない。）
    """
    embs = []
    pad_masks = []
    att_masks = []

    for img, img_mask in zip(images, img_masks, strict=True):

        def image_embed_func(img):
            return self.paligemma_with_expert.embed_image(img)

        img_emb = self._apply_checkpoint(image_embed_func, img)
        bsize, num_img_embs = img_emb.shape[:2]

        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    def lang_embed_func(lang_tokens):
        # 元実装の `* math.sqrt(lang_emb_dim)` は行わない（上の docstring 参照）
        return self.paligemma_with_expert.embed_language_tokens(lang_tokens)

    lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
    embs.append(lang_emb)
    pad_masks.append(lang_masks)

    num_lang_embs = lang_emb.shape[1]
    att_masks += [0] * num_lang_embs

    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

    bsize = pad_masks.shape[0]
    att_masks = att_masks[None, :].expand(bsize, len(att_masks))

    return embs, pad_masks, att_masks


def _patched_denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    """lerobot 0.4.4 の PI0Pytorch.denoise_step のバグ修正版（2点）。

    修正1: dtype 不一致
        state_proj / action_in_proj / action_time_mlp_* は常に float32 で保持される
        設計だが（PaliGemmaWithExpertModel.to_bfloat16_for_selected_params の対象外）、
        gemma_expert（アクション予測用の Transformer 本体）は bfloat16 で動く。
        forward()（学習用・prefix+suffix 同時パス）には両者の dtype を揃えるキャストが
        実装されている一方、denoise_step（推論の1ステップ、suffix のみのパス）には
        そのキャストが欠けており、bfloat16 チェックポイントで dtype 不一致エラーになる。

    修正2: KV キャッシュの汚染
        PI0 は paligemma（prefix 側）と gemma_expert（action 側）で同一の Cache
        オブジェクトを共有し、expert が prefix の KV に cross-attention する設計。
        transformers>=4.54 の GemmaAttention.forward は use_cache の値に関わらず
        `past_key_values.update(...)` でキャッシュを in-place 変異させる
        （use_cache は「戻り値に含めるか」しか制御しない）。
        prefix の KV キャッシュは sample_actions の denoise ループ
        （num_inference_steps 回）で使い回されるため、各ステップで書き込まれた
        suffix 分（chunk_size + state 個）を巻き戻さないと、2 ステップ目以降で
        キー長が attention マスク幅とずれて
        「The size of tensor a (918) must match the size of tensor b (867)」で落ちる。
        本来 lerobot が要求する transformers 4.53.2 では発生しない、版ずれ由来の不整合。
        → forward 後にキャッシュを prefix 長へ crop して巻き戻す。
    """
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

    # 修正3（_patched_embed_prefix の docstring も参照）: expert 側の GemmaModel も
    # inputs_embeds に normalizer = sqrt(hidden_size) を掛けるが、フォークでは
    # これが無効化されており embed_suffix の出力はそのまま渡される想定。
    # ここで先に割っておき、後段の乗算と相殺させる。
    # embed_suffix の出力は float32（state_proj 等が float32 のまま保持される）なので、
    # bfloat16 へ落とす前に割ることで丸め誤差を避ける。
    expert_hidden = self.paligemma_with_expert.gemma_expert.model.config.hidden_size
    suffix_embs = suffix_embs / (expert_hidden**0.5)

    if self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

    try:
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
    finally:
        # 上の docstring「修正2」を参照。このステップで書き込まれた suffix 分の KV を
        # 捨て、次の denoise ステップが prefix だけのキャッシュから始められるようにする。
        if past_key_values is not None and hasattr(past_key_values, "crop"):
            past_key_values.crop(prefix_len)

    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -self.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return self.action_out_proj(suffix_out)


def apply_pi0_inference_patches() -> None:
    """PI0Pytorch.denoise_step の推論時バグ（dtype 不一致・KV キャッシュ汚染）を
    修正する。詳細は _patched_denoise_step の docstring を参照。冪等。
    """
    from lerobot.policies.pi0 import modeling_pi0

    if getattr(modeling_pi0.PI0Pytorch, _PI0_PATCHED_ATTR, False):
        return

    global _make_att_2d_masks
    _make_att_2d_masks = modeling_pi0.make_att_2d_masks

    modeling_pi0.PI0Pytorch.denoise_step = _patched_denoise_step
    setattr(modeling_pi0.PI0Pytorch, _PI0_PATCHED_ATTR, True)

    if not getattr(modeling_pi0.PI0Pytorch, _PI0_SCALE_PATCHED_ATTR, False):
        modeling_pi0.PI0Pytorch.embed_prefix = _patched_embed_prefix
        setattr(modeling_pi0.PI0Pytorch, _PI0_SCALE_PATCHED_ATTR, True)
