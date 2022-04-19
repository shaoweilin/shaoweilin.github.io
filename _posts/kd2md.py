# python kd2md.py 

import re
from pathlib import Path

# front matter
# web links
# nested [[]]
# copy public
# copy images
# read more
# citations

# remove references, add biblio

# excerpts
# about.md
# singular.md
# find "<a id=" which do not come from biblio
  ## **<a id="theorem-convergence-of-biased-stochastic-approximation"></a>
  ## <a id="assumption-step-sizes-and-stop-time"></a>
  ## <a id="theorem-convergence-of-online-learning"></a>
  ## use https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html#targets-and-cross-referencing

SOURCE_PATH = Path("~/projects/shaoweilin.github.io/_posts").expanduser()
TARGET_PATH = Path("~/projects/shaoweilin.github.io/posts").expanduser()
POSTS = [
    "2012-07-13-studying-model-asymptotics-with-singular-learning-theory.md",
    "2014-08-02-boltzmann-machines-and-hierarchical-models.md",
    "2014-08-13-statistics-and-machine-learning.md",
    "2016-02-04-hashing.md",
    "2016-05-03-exercise-on-deep-neural-networks.md",
    "2016-05-03-exercise-on-sparse-autoencoders.md",
    "2017-05-08-artificial-general-intelligence-for-the-internet-of-things.md",
    "2018-05-26-machine-reasoning-and-deep-spiking-networks.md",
    "2020-05-21-logical-frameworks.md",
    "2020-05-26-directed-spaces-and-types.md",
    "2020-07-23-adjunctions.md",
    "2020-08-07-processes-and-variety-maximization.md",
    "2020-08-28-motivic-information-path-integrals-and-spiking-networks.md",
    "2020-09-08-building-foundations-of-information-theory-on-relative-information.md",
    "2020-09-18-conditional-relative-information-and-its-axiomatizations.md",
    "2020-10-05-zeta-functions-mellin-transforms-and-the-gelfand-leray-form.md",
    "2020-10-07-motivic-relative-information.md",
    "2020-10-14-path-integrals-and-continuous-time-markov-chains.md",
    "2020-10-23-machine-learning-with-relative-information.md",
    "2020-12-01-biased-stochastic-approximation.md",
    "2021-03-21-process-learning-with-relative-information.md",
    "2021-03-22-relative-inference-with-mutable-processes.md",
    "2021-03-23-biased-stochastic-approximation-with-mutable-processes.md",
    "2021-04-22-proofs-as-programs-challenges-and-strategies-for-program-synthesis.md",
    "2021-05-10-path-integrals-and-the-dyson-formula.md",
    "2021-06-01-convergence-of-biased-stochastic-approximation.md",
    "2021-06-05-spiking-neural-networks.md",
    "2021-08-01-information-type-theory.md",
    "2021-09-09-all-you-need-is-relative-information.md",
    "2022-01-22-information-topos-theory-motivation.md"
]

LINKS = dict([(post[11:-3],post[:-3]) for post in POSTS])

BIBLIO = {
    "L06":"luo1992unifying",
    "HHP93":"harper1993framework",
    "H12":"harper2012notes",
    "H21":"harper2020semantic",
    "n20":"nlab2020logicalframework",
    "BF14":"baez2014bayesian",
    "C17":"chodrow2017divergence",
    "G11":"gray2011entropy",
    "J57":"jaynes1957information",
    "B11":"baez2011characterization",
    "AGV88":"arnold1988elementary",
    "B05":"baez2005week",
    "D91":"deninger1991gamma",
    "M95":"manin1995lectures",
    "M10":"marcolli2010feynman",
    "W09":"watanabe2009algebraic",
    "H05":"hales2005motivic",
    "M19":"marcolli2019motivic",
    "P02":"poonen2002grothendieck",
    "S09":"scanlon2009motivic",
    "AM11":"albeverio2011path",
    "AMB17":"albeverio2017probabilistic",
    "GZ02":"gill2002foundations",
    "H14":"hairer2014theory",
    "I16":"inahama2019rough",
    "A95":"amari1995information",
    "BKM17":"blei2017variational",
    "DLR77":"dempster1977maximum",
    "JGJS99":"jordan1999introduction",
    "B98":"bottou1998online",
    "CCG11":"cattiaux2012central",
    "KMMW19":"karimi2019non",
    "LR15":"li2015stochastic",
    "O13":"oksendal2013stochastic",
    "L92":"leroux1992maximum",
    "BB01":"baxter2001infinite",
    "RG14":"rezende2014stochastic",
    "PBR20":"pozzi2020attention",
    "B13":"brown2013iterated",
    "G17":"gill2017feynman",
    "N21":"nlab2021dysonformula",
    "F10":"friston2010free",
    "HH52":"hodgkin1952quantitative",
    "KB94":"kuo1994na+",
    "LCHL20":"liu2020biologically",
    "LL19":"lindgren2019quantum",
    "PTBG06":"pfister2006optimal",
    "R21":"raviv2018genius",
    "H00":"heeger2000poisson",
    "L19":"leinster2019short",
    "MS09":"maclagan2015introduction",
    "A18":"awodey2018natural",
    "B20a":"baez2020topos2",
    "B20b":"baez2020topos5"
}

# duplicates ['G11', 'W09', 'H14', 'KMMW19', 'KMMW19', 'KMMW19', 'RG14', 'B11', 'BF14', 'BF14']


def fix_front_matter(str,post):
    # get front matter
    re_front_matter = "---\n([\s\S]*?)---"
    front_matter_match = re.search(re_front_matter,str)
    front_matter = front_matter_match.group(1)
    front_matter_start, front_matter_stop = front_matter_match.span()
    assert(front_matter_start==0)

    # get post title
    re_key_value = "([^:]*):[ ]*([^\n]*)\n"
    key_value_match = re.findall(re_key_value,front_matter)
    post_title = None
    for key, value in key_value_match:
        if key=="layout":
            assert(value=="post")
        elif key=="title":
            post_title = value
        elif key=="excerpt_separator":
            assert(value=="<!--more-->")
        else:
            raise ValueError(f"Key {key}  not recognized")
    if post_title is None:
        raise ValueError("Post title not found")
    
    # get post date
    post_date = post[:10]

    # replace front matter
    newstr  = f"---\n"
    newstr += f"date: {post_date}\n"
    newstr += f"excerpts: 2\n"
    newstr += f"---\n\n"
    newstr += f"# {post_title}"
    newstr += str[front_matter_stop:]
    return newstr

def all_links(str):
    EXPECTING_SQ_BRAC = 0
    OUTER_SQ_BRAC_HEAD = 1
    OUTER_SQ_BRAC_BODY = 2
    OUTER_SQ_BRAC_TAIL = 3
    INNER_SQ_BRAC_HEAD_BODY = 4
    INNER_SQ_BRAC_TAIL = 5
    RD_BRAC_HEAD_BODY = 6
    INNER_RD_BRAC = 7
    LATEX_HEAD = 8
    LATEX_BODY = 9
    LATEX_TAIL = 10
    
    state = EXPECTING_SQ_BRAC
    prev_not_esc = True
    outer_body = ""
    inner_body = ""
    round_body = ""
    inner_ref = ""
    inner_tag = ""
    inner_bib = ""
    for pos, char in enumerate(str):
        if state == EXPECTING_SQ_BRAC:
            if char == "[" and prev_not_esc:
                state = OUTER_SQ_BRAC_HEAD
                start = pos
                outer_body = ""
            elif char == "$":
                state = LATEX_HEAD
        elif state == OUTER_SQ_BRAC_HEAD:
            if char == "[":
                state = INNER_SQ_BRAC_HEAD_BODY
                inner_body = ""
            elif char == "]":
                state = OUTER_SQ_BRAC_TAIL
            else:
                outer_body += char
                state = OUTER_SQ_BRAC_BODY
        elif state == OUTER_SQ_BRAC_BODY:
            if char == "]":
                state = OUTER_SQ_BRAC_TAIL
            elif char == "[":
                raise ValueError(f"unexpected {char} when OUTER_SQ_BRAC_BODY at pos {pos}")
            else:
                outer_body += char
        elif state == OUTER_SQ_BRAC_TAIL:
            if char == "(":
                state = RD_BRAC_HEAD_BODY
                round_body = ""
            else:
                state = EXPECTING_SQ_BRAC
        elif state == INNER_SQ_BRAC_HEAD_BODY:
            if char == "]":
                state = INNER_SQ_BRAC_TAIL
            elif char == "[":
                raise ValueError(f"unexpected {char} when INNER_SQ_BRAC_HEAD_BODY at pos {pos}")
            else:
                inner_body += char
        elif state == INNER_SQ_BRAC_TAIL:
            if char == "]":
                state = OUTER_SQ_BRAC_TAIL
                inner_body_match = re.search("([^ ]*)([\s\S]*)", inner_body)
                inner_ref = inner_body_match.group(1) 
                inner_tag = inner_body_match.group(2)
                inner_bib = ""
                if start >= 4:
                    if str[start-4:start]=="</a>": 
                        inner_bib = inner_ref
                        inner_ref = ""
                        inner_tag = ""
            else:
                raise ValueError(f"unexpected {char} when INNER_SQ_BRAC_TAIL at pos {pos}")
        elif state == RD_BRAC_HEAD_BODY:
            if char == ")":
                state = EXPECTING_SQ_BRAC
                stop = pos+1
                yield (outer_body,inner_ref,inner_tag,inner_bib,round_body,start,stop)
                outer_body = ""
                inner_ref = ""
                inner_tag = ""
                inner_bib = ""
                round_body = ""
            elif char == "(":
                state = INNER_RD_BRAC
                round_body += char
            else: 
                round_body += char
        elif state == INNER_RD_BRAC:
            if char == ")":
                state = RD_BRAC_HEAD_BODY
            elif char == "(":                 
                raise ValueError(f"unexpected {char} when INNER_RD_BRAC at pos {pos}")
            round_body += char
        elif state == LATEX_HEAD:
            if char != "$":
                state = LATEX_BODY
        elif state == LATEX_BODY:
            if char == "$":
                state = LATEX_TAIL
        elif state == LATEX_TAIL:
            if char == "[":
                state = OUTER_SQ_BRAC_HEAD
                start = pos
                outer_body = ""
            elif char != "$":
                state = EXPECTING_SQ_BRAC
        else:
            raise ValueError(f"unexpected state {state}")
        if char == "\\":
            prev_not_esc = False
        else:
            prev_not_esc = True

def fix_links(str):
    newstr = ""
    prev_stop = 0
    for text,ref,tag,bib,link,start,stop in all_links(str):
        #print(text,ref,tag,link,start,stop)
        newstr += str[prev_stop:start]
        if text != "":
            if link[:29] == "https://shaoweilin.github.io/" :
                re_shaowei = "https://shaoweilin.github.io/([\s\S]*?)/([\s\S]*)"
                shaowei_match =  re.search(re_shaowei,link)
                subdomain = shaowei_match.group(1)
                sitetag = shaowei_match.group(2)
                if subdomain in LINKS:
                    newstr += f"[{text}]({LINKS[subdomain]}/{sitetag})"
                    #print(f"[{text}]({LINKS[subdomain]}/{sitetag})")
                elif subdomain == "singular":
                    newstr += f"[{text}](../{subdomain}/{sitetag})"
                    print(f"[{text}](../{subdomain}/{sitetag})")
                elif subdomain == "images":
                    newstr += f"[{text}](../{subdomain}/{sitetag})"
                    print(f"[{text}](../{subdomain}/{sitetag})")
                else:
                    raise ValueError(f"Link {link} not understood")
            else:
                newstr += f"[{text}]({link})"
        elif ref != "":
            newstr += f"{{cite}}`{BIBLIO[ref]}`"
        else:
            assert(bib!="")
        prev_stop = stop
    newstr += str[prev_stop:-1]
    return newstr

def fix_excerpt_separator(str):
    return re.sub("<!--more-->[ ]*\n","",str)

def fix_references(str):
    re_references = "##[ ]*References[ ]*\n[\s\S]*"
    references_match = re.search(re_references,str)
    if references_match is not None:
        references_start, _ = references_match.span()
        newstr = str[:references_start]
        newstr += "## References\n\n"
        newstr += "```{bibliography}\n:filter: docname in docnames\n```"
        return newstr
    else:
        return str

def find_latex(str):
    padded_str = "\n"+str+"\n"
    outer_match = re.search("[^ ][ ]*\$\$[^\$]*\$\$[ ]*[^ ]",padded_str)
    if outer_match is None:
        return None
    else:
        outer_group = outer_match.group()
        outer_start, outer_stop = outer_match.span()
        inner_match = re.search("\$\$[^\$]*\$\$", outer_group)
        char_start = outer_group[0]
        char_stop = outer_group[-1]
        is_display = (char_start=="\n" and char_stop=="\n")
        inner_start, inner_stop = inner_match.span()
        latex_start = outer_start + inner_start
        latex_stop = outer_start + inner_stop
        return (is_display, latex_start-1, latex_stop-1)

def fix_latex(str):
    newstr = ""
    start = 0
    while True:
        str = str[start:]
        results = find_latex(str)
        if results is None:
            newstr += str
            break
        else:
            is_display, latex_start, latex_end = results
            start = latex_end
            if is_display:
                newstr += str[:start]
            else:
                newstr += str[:latex_start]
                newstr += str[(latex_start+1):(latex_end-1)]
    return newstr

def find_duplicate_bib():
    bibs = {}
    dups = []
    for post in POSTS:
        print(post)
        kdfile = SOURCE_PATH/post
        mdfile = TARGET_PATH/post
        with open(kdfile,"r") as f:
            str = f.read()
            for text,ref,tag,bib,link,start,stop in all_links(str):
                if bib != "":
                    if bib in bibs.keys():
                        print("    DUP",bib,bibs[bib])
                        dups += [bib]
                    else:
                        bibs[bib] = post
    print("duplicate bib\n",dups)

def find_cites_with_tags():
    for post in POSTS:
        print(post)
        kdfile = SOURCE_PATH/post
        mdfile = TARGET_PATH/post
        with open(kdfile,"r") as f:
            str = f.read()
            for text,ref,tag,bib,link,start,stop in all_links(str):
                if tag != "":
                    print(ref,tag,bib)

def jekyll2myst(post):
    kdfile = SOURCE_PATH/post
    mdfile = TARGET_PATH/post
    with open(kdfile,"r") as f:
        str = f.read()
        str = fix_front_matter(str,post)
        str = fix_excerpt_separator(str)
        str = fix_links(str)
        str = fix_latex(str)
        str = fix_references(str)
    with open(mdfile,"w") as f:
        f.write(str)

if __name__ == "__main__":
    for post in POSTS:
        jekyll2myst(post)
    #find_duplicate_bib()
    #find_cites_with_tags()
