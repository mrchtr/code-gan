from utils.Tokenizer import SentencepieceResolver

special_tokens = [
    '<BOF>',
    '<EOF>',
    '<COMMENT>',
    '<STR_LIT>',
    '<INT_LIT>'
]
resolver = SentencepieceResolver(path ="../demo_code/out_jokes.py", vocab_size=2000, name="tokenizer", special_tokens=special_tokens)
resolver.train()
resolver.load()
print(resolver.encode("<BOF> hello world \n <EOF>"))
print(resolver.decode(resolver.encode("hello world")))

print(resolver.tokenizer.piece_to_id('<BOF>'))
print(resolver.encode("<EOF>"))