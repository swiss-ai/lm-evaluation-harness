import functools


def doc_to_target(doc):
    return doc["content"]


def process_results(doc, results):
    (loglikelihood,) = results
    _bytes = len(doc["content"].encode("utf-8"))
    return {
        "bits_per_byte": (loglikelihood, _bytes),
        "byte_perplexity": (loglikelihood, _bytes),
    }


def _filter_category(dataset, category):
    return dataset.filter(lambda doc: doc["category"] == category)


process_arxiv_cs = functools.partial(_filter_category, category="arxiv_cs")
process_arxiv_math = functools.partial(_filter_category, category="arxiv_math")
process_arxiv_physics = functools.partial(_filter_category, category="arxiv_physics")
process_arxiv_other = functools.partial(_filter_category, category="arxiv_other")
process_biorxiv_all = functools.partial(_filter_category, category="biorxiv_all")
process_bbc_news = functools.partial(_filter_category, category="bbc_news")
process_wikipedia_english = functools.partial(
    _filter_category, category="wikipedia_english"
)
process_wikipedia_nonenglish = functools.partial(
    _filter_category, category="wikipedia_nonenglish"
)
process_ao3_english = functools.partial(_filter_category, category="ao3_english")
process_ao3_nonenglish = functools.partial(_filter_category, category="ao3_nonenglish")
process_github_cpp = functools.partial(_filter_category, category="github_cpp")
process_github_javascript = functools.partial(
    _filter_category, category="github_javascript"
)
process_github_markdown = functools.partial(
    _filter_category, category="github_markdown"
)
process_github_python = functools.partial(_filter_category, category="github_python")
process_github_other = functools.partial(_filter_category, category="github_other")
