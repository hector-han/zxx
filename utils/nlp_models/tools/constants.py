

class VocabParams(object):
    unk     = "<unk>"
    pad     = "<pad>"
    go      = "<go>"
    eos     = "<eos>"
    space   = "<space>"
    split   = "<split>"
    label_prefix = "__label__"
    unk_label = "__label__unknown"

ch2en = {
    '正': 'POSITIVE',
    '负': 'NEGATIVE',
    '中': 'NEUTRAL',
}