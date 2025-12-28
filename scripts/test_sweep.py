import torch
from diffusers import ZImagePipeline
import os

# 1. 모델 준비 (기존 코드와 동일)
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16, 
    use_safetensors=True, 
    # trust_remote_code=True
)
pipe.to("cuda")
# pipe.vae.to(dtype=torch.float32)

# 2. 테스트 환경 설정
prompt = "A high-tech mechanical keyboard with RGB lighting, macro photography, detailed keycaps"
steps_to_test = [1, 4, 8, 16]      # 망치질 횟수 테스트
guidance_to_test = [0.0, 1.5, 5.0] # 잔소리 강도 테스트

os.makedirs("sweep_results", exist_ok=True)

# 3. 중첩 루프로 모든 조합 테스트 (Grid Search)
print("🚀 매트릭스 테스트 시작...")
for steps in steps_to_test:
    for gs in guidance_to_test:
        filename = f"sweep_results/step{steps}_gs{gs}.png"
        print(f"🎬 생성 중: Steps {steps}, Guidance {gs} -> {filename}")
        
        # 시드 고정 (변수 차이를 명확히 보기 위해 동일한 초기 노이즈 사용)
        generator = torch.Generator("cuda").manual_seed(42)
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=gs,
            generator=generator
        ).images[0]
        
        image.save(filename)

print("✅ 테스트 완료! 'sweep_results' 폴더의 파일들을 비교해 보세요.")