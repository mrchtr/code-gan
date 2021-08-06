from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")


test = ["<s>", "from", "bootstrap", "import", "Bootstrap", "<EOL>", "from", "fund", "import", "InstantPaymentNotificationHandler", "<EOL>", "from", "fund", "import", "ThankYouHandler", "<EOL>", "from", "view", "import", "*", "<EOL>", "mapping", "=", "[", "(", "<EOL>", "r'/'", ",", "<EOL>", "Index", "<EOL>", ")", ",", "(", "<EOL>", "r'/ipn'", ",", "<EOL>", "InstantPaymentNotificationHandler", "<EOL>", ")", ",", "(", "<EOL>", "r'/thank-you'", ",", "<EOL>", "ThankYouHandler", "<EOL>", ")", ",", "(", "<EOL>", "r'/about\\/?'", ",", "<EOL>", "About", "<EOL>", ")", ",", "(", "<EOL>", "r'/guide\\/?'", ",", "<EOL>", "Guide", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Download", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Standards", "<EOL>", ")", ",", "(", "<EOL>", "r'/community\\/?'", ",", "<EOL>", "Community", "<EOL>", ")", ",", "(", "<EOL>", "r'/news\\/?'", ",", "<EOL>", "News", "<EOL>", ")", ",", "(", "<EOL>", "r'/support\\/?'", ",", "<EOL>", "Support", "<EOL>", ")", ",", "(", "<EOL>", "r'/contact\\/?'", ",", "<EOL>", "Contact", "<EOL>", ")", ",", "(", "<EOL>", "r'/press\\/?'", ",", "<EOL>", "Press", "<EOL>", ")", ",", "(", "<EOL>", "r'/legal/terms'", ",", "<EOL>", "Terms", "<EOL>", ")", ",", "(", "<EOL>", "r'/library\\/?'", ",", "<EOL>", "Library", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Library", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Library", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Users", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "User", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Design", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Design", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Design", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Design", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Design", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "RedirectSuccess", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "RedirectError", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "RedirectAfterDelete", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Moderate", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Bootstrap", "<EOL>", ")", ",", "(", "<EOL>", "r'/activity'", ",", "<EOL>", "ActivityScreen", "<EOL>", ")", ",", "(", "<EOL>", "r'/txns'", ",", "<EOL>", "TxnList", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Base64Blob", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "Base64Blob", "<EOL>", ")", ",", "(", "<EOL>", "r''", ",", "<EOL>", "MessageStrings", "<EOL>", ")", ",", "(", "<EOL>", "r'/.*'", ",", "<EOL>", "NotFound", "<EOL>", ")", "<EOL>", "]", "</s>"]

dataset = load_dataset("code_x_glue_cc_code_completion_line", "python", split='train')
iterator = iter(dataset)
for i in range(len(dataset)):
    row = next(iterator)
    encoded = tokenizer.encode(row['input'])
    print(encoded)
    print(tokenizer.decode(tokenizer.encode(row['input'])))
    input = ' '.join(row['input'])
    print(row)
