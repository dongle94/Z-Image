import torch
from diffusers import ZImagePipeline
import time

class ZImageArtist:
    def __init__(self, device="cuda"):
        self.device = device
        self.model_id = "Tongyi-MAI/Z-Image-Turbo"
        self.pipe = self._load_model()
        
    def _load_model(self):
        print(f"📡 모델 로딩 시작: {self.model_id}")
        pipe = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16, # 개발자님이 찾아낸 최적 정밀도
            use_safetensors=True,
            trust_remote_code=True
        )
        pipe.to(self.device)
        pipe.vae.to(dtype=torch.float32) # 검은 화면 방지 핵심 로직
        return pipe

    def generate(self, prompt, steps=8, gs=0.0, seed=None):
        generator = torch.Generator(self.device)
        if seed is not None:
            generator.manual_seed(seed)
            
        start_time = time.time()
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=gs,
            generator=generator
        ).images[0]
        
        elapsed = time.time() - start_time
        print(f"✅ 생성 완료 ({elapsed:.2f}초)")
        return image