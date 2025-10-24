import torch
from sentence_transformers import SentenceTransformer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('detected device:', device)

    # Load small model to keep runtime short
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Inspect a parameter before moving
    try:
        p = next(model.parameters())
        print('param before .to device:', p.device, 'dtype:', p.dtype)
    except StopIteration:
        print('model has no parameters')

    # Move model to device if possible
    try:
        model = model.to(device)
        p = next(model.parameters())
        print('param after .to device:', p.device, 'dtype:', p.dtype)
    except Exception as e:
        print('model.to failed or not applicable:', e)

    # Tokenize a small sample and show tensor devices before/after moving
    texts = ['Hello world', 'Testing device placement']
    try:
        toks = model.tokenize(texts)
        for k, v in toks.items():
            print('token key', k, 'type', type(v), 'device', getattr(v, 'device', None), 'dtype', getattr(v, 'dtype', None))
        moved = {}
        for k, v in toks.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device, non_blocking=True)
            else:
                moved[k] = v
        for k, v in moved.items():
            print('moved key', k, 'device', getattr(v, 'device', None), 'dtype', getattr(v, 'dtype', None))
    except Exception as e:
        print('tokenize/move failed:', e)

if __name__ == '__main__':
    main()
