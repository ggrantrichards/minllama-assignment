import torch
from llama import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sanity_data = torch.load("./sanity_check.data")

sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])

llama = load_pretrained("stories42M.pt")
llama.eval()
with torch.no_grad():
    logits, hidden_states = llama(sent_ids)    
    logits = sanity_data["logits"].to(logits.device).type_as(logits)
    hidden_states = sanity_data["hidden_states"].to(hidden_states.device).type_as(hidden_states)
    print(f"Logits Shape: {logits.shape}")
    print(f"Target Shape: {sanity_data['logits'].shape}")
    diff = torch.abs(logits - sanity_data["logits"])
    print(f"Max difference in Logits: {diff.max().item()}")
    print(f"Mean difference in Logits: {diff.mean().item()}")
    h_diff = torch.abs(hidden_states - sanity_data["hidden_states"])
    print(f"Max difference in Hidden States: {h_diff.max().item()}")
    assert torch.allclose(logits, sanity_data["logits"], atol=1e-5, rtol=1e-3)
    assert torch.allclose(hidden_states, sanity_data["hidden_states"], atol=1e-5, rtol=1e-3)
    print("Your Llama implementation is correct!")