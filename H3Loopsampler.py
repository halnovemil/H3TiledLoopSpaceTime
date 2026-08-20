import copy
import torch
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview
from comfy.nested_tensor import NestedTensor


class H3LoopingSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "temporal_tile_size": (
                    "INT",
                    {
                        "default": 81,
                        "min": 17,
                        "max": 257,
                        "step": 4,
                        "tooltip": "Tamanho do tile temporal em frames de latente (vídeo)",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 17,
                        "min": 5,
                        "max": 65,
                        "step": 4,
                    },
                ),
                "temporal_overlap_strength": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 4}),
                "vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 4}),
                "spatial_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 32,
                    },
                ),
            },
            "optional": {
                "adain_factor": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/H3"
    DESCRIPTION = "H3 Looping / Tiled Sampler com output + denoised_output (compatível com SplitSigmas) ComfyGuy9000"

    def _is_nested(self, samples):
        return isinstance(samples, NestedTensor) or getattr(samples, "is_nested", False)

    def _get_tensors(self, samples):
        if self._is_nested(samples):
            if hasattr(samples, "tensors"):
                return list(samples.tensors)
            return list(samples.unbind())
        return [samples]

    def _make_nested(self, tensors):
        if len(tensors) == 1:
            return tensors[0]
        return NestedTensor(tensors)

    def _get_video(self, samples):
        return self._get_tensors(samples)[0]

    def _slice_video_temporal(self, video, start, end):
        return video[:, :, start:end].clone()

    def _slice_video_spatial(self, video, v_start, v_end, h_start, h_end):
        return video[:, :, :, v_start:v_end, h_start:h_end].clone()

    def _create_spatial_weights(self, shape, v, h, vertical_tiles, horizontal_tiles, spatial_overlap, device, dtype):
        weights = torch.ones(shape, device=device, dtype=dtype)
        if spatial_overlap > 0:
            if h > 0:
                blend = torch.linspace(0, 1, spatial_overlap, device=device, dtype=dtype)
                weights[..., :spatial_overlap] *= blend.view(1, 1, 1, 1, -1)
            if h < horizontal_tiles - 1:
                blend = torch.linspace(1, 0, spatial_overlap, device=device, dtype=dtype)
                weights[..., -spatial_overlap:] *= blend.view(1, 1, 1, 1, -1)
            if v > 0:
                blend = torch.linspace(0, 1, spatial_overlap, device=device, dtype=dtype)
                weights[..., :spatial_overlap, :] *= blend.view(1, 1, 1, -1, 1)
            if v < vertical_tiles - 1:
                blend = torch.linspace(1, 0, spatial_overlap, device=device, dtype=dtype)
                weights[..., -spatial_overlap:, :] *= blend.view(1, 1, 1, -1, 1)
        return weights

    def _adain(self, source, target, factor):
        if factor <= 0.0:
            return source
        src_mean = source.mean(dim=(2, 3, 4), keepdim=True)
        src_std = source.std(dim=(2, 3, 4), keepdim=True) + 1e-5
        tgt_mean = target.mean(dim=(2, 3, 4), keepdim=True)
        tgt_std = target.std(dim=(2, 3, 4), keepdim=True) + 1e-5
        normalized = (source - src_mean) / src_std
        stylized = normalized * tgt_std + tgt_mean
        return source * (1.0 - factor) + stylized * factor

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        temporal_tile_size,
        temporal_overlap,
        temporal_overlap_strength,
        horizontal_tiles,
        vertical_tiles,
        spatial_overlap,
        adain_factor=0.15,
    ):
        original_latent = latent_image
        samples = latent_image["samples"]

        video = self._get_video(samples)
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(video.shape)}")

        B, C, T, H, W = video.shape
        print(f"\n========== H3LoopingSampler ComfyGuy9000==========")
        print(f"Input video latent: {video.shape}")
        print(f"Tiles: {vertical_tiles}x{horizontal_tiles} | spatial_overlap={spatial_overlap}")
        print(f"Temporal tile={temporal_tile_size} | overlap={temporal_overlap}")

        original_tensors = self._get_tensors(samples)
        has_audio = len(original_tensors) > 1
        full_audio = original_tensors[1] if has_audio else None

        temporal_tile_size = min(temporal_tile_size, T)
        temporal_overlap = min(temporal_overlap, max(4, temporal_tile_size - 4))

        if vertical_tiles > 1:
            base_tile_h = (H + (vertical_tiles - 1) * spatial_overlap) // vertical_tiles
        else:
            base_tile_h = H
        if horizontal_tiles > 1:
            base_tile_w = (W + (horizontal_tiles - 1) * spatial_overlap) // horizontal_tiles
        else:
            base_tile_w = W

        print(f"Base tile size (latent): {base_tile_h} x {base_tile_w}")

        final_video = None
        final_denoised_video = None
        weights = None
        first_seed = noise.seed
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        tile_count = 0
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                v_start = v * (base_tile_h - spatial_overlap)
                h_start = h * (base_tile_w - spatial_overlap)
                v_end = min(v_start + base_tile_h, H) if v < vertical_tiles - 1 else H
                h_end = min(h_start + base_tile_w, W) if h < horizontal_tiles - 1 else W

                tile_count += 1
                print(f"\n>>> Spatial tile {tile_count}/{vertical_tiles*horizontal_tiles} ({v},{h})")
                print(f"    H[{v_start}:{v_end}] W[{h_start}:{h_end}]")

                spatial_video = self._slice_video_spatial(video, v_start, v_end, h_start, h_end)

                tile_out_video = None
                tile_denoised_video = None
                first_chunk_ref = None

                step = max(1, temporal_tile_size - temporal_overlap)
                starts = list(range(0, max(1, T - temporal_overlap), step))

                for i, start in enumerate(starts):
                    end = min(start + temporal_tile_size, T)
                    print(f"    Temporal chunk {i}: [{start}:{end}]")

                    chunk_video = self._slice_video_temporal(spatial_video, start, end)

                    if has_audio:
                        chunk_samples = self._make_nested([chunk_video, full_audio])
                    else:
                        chunk_samples = chunk_video

                    chunk_latent = {"samples": chunk_samples}
                    if "noise_mask" in latent_image:
                        chunk_latent["noise_mask"] = latent_image["noise_mask"]

                    noise.seed = first_seed + start * (vertical_tiles * horizontal_tiles) + v * horizontal_tiles + h

                    # === Captura do x0 (denoised) ===
                    x0_output = {}
                    callback = latent_preview.prepare_callback(
                        guider.model_patcher, sigmas.shape[-1] - 1, x0_output
                    )

                    noise_mask = chunk_latent.get("noise_mask", None)

                    out_samples = guider.sample(
                        noise.generate_noise(chunk_latent),
                        chunk_samples,
                        sampler,
                        sigmas,
                        denoise_mask=noise_mask,
                        callback=callback,
                        disable_pbar=disable_pbar,
                        seed=noise.seed,
                    )

                    out_samples = out_samples.to(comfy.model_management.intermediate_device())
                    chunk_out_video = self._get_video(out_samples)

                    # Pega a versão denoised (x0) se disponível
                    if "x0" in x0_output:
                        x0 = x0_output["x0"]
                        if self._is_nested(out_samples) and not self._is_nested(x0):
                            # Tenta reconstruir NestedTensor se necessário
                            try:
                                latent_shapes = [t.shape for t in self._get_tensors(out_samples)]
                                x0 = NestedTensor(comfy.utils.unpack_latents(x0, latent_shapes))
                            except:
                                pass
                        chunk_denoised_video = self._get_video(x0)
                        # Aplica process_latent_out se possível
                        try:
                            chunk_denoised_video = guider.model_patcher.model.process_latent_out(chunk_denoised_video.cpu()).to(chunk_out_video.device)
                        except:
                            chunk_denoised_video = chunk_denoised_video.to(chunk_out_video.device)
                    else:
                        chunk_denoised_video = chunk_out_video

                    # AdaIN (aplica nos dois)
                    if first_chunk_ref is None:
                        first_chunk_ref = chunk_out_video.detach()
                    else:
                        ref = first_chunk_ref
                        if ref.shape[2] != chunk_out_video.shape[2]:
                            ref = first_chunk_ref[:, :, :1].expand_as(chunk_out_video)
                        else:
                            ref = first_chunk_ref[:, :, :chunk_out_video.shape[2]]
                        chunk_out_video = self._adain(chunk_out_video, ref, adain_factor)
                        chunk_denoised_video = self._adain(chunk_denoised_video, ref, adain_factor)

                    # Blend temporal - output normal
                    if tile_out_video is None:
                        tile_out_video = chunk_out_video
                        tile_denoised_video = chunk_denoised_video
                    else:
                        overlap = temporal_overlap
                        if overlap > 0 and tile_out_video.shape[2] >= overlap:
                            # Blend output normal
                            prev = tile_out_video[:, :, -overlap:]
                            new = chunk_out_video[:, :, :overlap]
                            alpha = torch.linspace(1.0, 0.0, overlap, device=tile_out_video.device, dtype=tile_out_video.dtype).view(1, 1, -1, 1, 1)
                            blended = prev * alpha + new * (1.0 - alpha) * temporal_overlap_strength + new * (1.0 - temporal_overlap_strength) * (1.0 - alpha)
                            tile_out_video = torch.cat([tile_out_video[:, :, :-overlap], blended, chunk_out_video[:, :, overlap:]], dim=2)

                            # Blend denoised
                            prev_d = tile_denoised_video[:, :, -overlap:]
                            new_d = chunk_denoised_video[:, :, :overlap]
                            blended_d = prev_d * alpha + new_d * (1.0 - alpha) * temporal_overlap_strength + new_d * (1.0 - temporal_overlap_strength) * (1.0 - alpha)
                            tile_denoised_video = torch.cat([tile_denoised_video[:, :, :-overlap], blended_d, chunk_denoised_video[:, :, overlap:]], dim=2)
                        else:
                            tile_out_video = torch.cat([tile_out_video, chunk_out_video], dim=2)
                            tile_denoised_video = torch.cat([tile_denoised_video, chunk_denoised_video], dim=2)

                # Acumula spatial
                if final_video is None:
                    out_T = tile_out_video.shape[2]
                    final_video = torch.zeros(B, C, out_T, H, W, device=tile_out_video.device, dtype=tile_out_video.dtype)
                    final_denoised_video = torch.zeros_like(final_video)
                    weights = torch.zeros_like(final_video)

                if tile_out_video.shape[2] != final_video.shape[2]:
                    if tile_out_video.shape[2] > final_video.shape[2]:
                        tile_out_video = tile_out_video[:, :, :final_video.shape[2]]
                        tile_denoised_video = tile_denoised_video[:, :, :final_video.shape[2]]
                    else:
                        pad = final_video.shape[2] - tile_out_video.shape[2]
                        tile_out_video = torch.nn.functional.pad(tile_out_video, (0, 0, 0, 0, 0, pad))
                        tile_denoised_video = torch.nn.functional.pad(tile_denoised_video, (0, 0, 0, 0, 0, pad))

                w = self._create_spatial_weights(
                    tile_out_video.shape, v, h, vertical_tiles, horizontal_tiles,
                    spatial_overlap, tile_out_video.device, tile_out_video.dtype
                )

                tile_out_video = tile_out_video.to(final_video.device)
                tile_denoised_video = tile_denoised_video.to(final_video.device)
                w = w.to(final_video.device)

                final_video[:, :, :, v_start:v_end, h_start:h_end] += tile_out_video * w
                final_denoised_video[:, :, :, v_start:v_end, h_start:h_end] += tile_denoised_video * w
                weights[:, :, :, v_start:v_end, h_start:h_end] += w

        final_video = final_video / (weights + 1e-8)
        final_denoised_video = final_denoised_video / (weights + 1e-8)
        noise.seed = first_seed

        # Monta NestedTensor para as duas saídas
        def make_output_latent(video_tensor):
            out_tensors = [video_tensor]
            if has_audio:
                out_tensors.append(full_audio.to(video_tensor.device))
            out_samples = self._make_nested(out_tensors)
            out_latent = copy.deepcopy(original_latent)
            out_latent["samples"] = out_samples
            return out_latent

        output_latent = make_output_latent(final_video)
        denoised_latent = make_output_latent(final_denoised_video)

        print(f"\n[H3LoopingSampler] Final video shape: {final_video.shape}")
        print(f"Total spatial tiles: {tile_count}")
        print("Saídas: output + denoised_output")
        print("========================================\n")

        return (output_latent, denoised_latent)


NODE_CLASS_MAPPINGS = {
    "H3LoopingSampler": H3LoopingSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LoopingSampler_ComfyGuy9000": "H3 Looping / Tiled Sampler"
}