# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as plt

#------------------------------------------------------------------------------
# Create Dictionary Data Structure
dataset = pd.read_csv("E:/Final Year/Text Simplification/Code/new_dictionary.csv")
#dataset =
#dataset=dataset.dropna()
dictionary=[[] for x in range(4274)]

irregular_verbs = pd.read_csv("E:/Final Year/Text Simplification/Code/Irregular_verbs.csv")

count=0
for index,row in dataset.iterrows():
    if count>=4274:
        break
    word=row["words"]
    meanings= row["meanings"].split("; ")
    dictionary[count].append(word)
    for x in meanings:
        dictionary[count].append(x)
    count+=1

#-----------------------------------------------------------------------------
# Pre-Processing
import nltk,re,pprint
from nltk import sent_tokenize,word_tokenize
f=open('document.txt')
h=open('document1.txt','w+')
new1=[]
#raw=f.read()
#raw='He had already inadvertently caused her enough problems.'
raw='It is important to read through contracts before you sign them because some things are not overtly stated.'
#raw='At home Kruger continued as obdurate as ever.'
#raw='Kurt hoped he could palliate his wife’s anger by buying her flowers.'
#raw='He hated her and was forever sundered from her.'
#raw='Our lawyers will niggle over every little detail of the case.'
#raw='In todays computer world, a floppy disk is an anachronism.'
#raw='The handprint was a bright red blotch on his pale face as he stared down at her.'
#raw='In a brief epilogue, the apostle justifies himself for having thus addressed the Roman Christians.'
#raw='his functionary is first formally mentioned under Leo X.(1513-1521) in the proceedings in connexion with the canonization of St Lorenzo Giustiniani.'
#raw=''

lesk_raw=raw.replace(',','')
lesk_raw=lesk_raw.replace('.','')

#------------------------------------------------------------------------------
# Tokenization and POS tagging
#for line in f:
#   print(line.strip())
words=word_tokenize(raw)
ls=nltk.pos_tag(words)
ls1=[]
ls2=[]
words_initial=[]
words_final=[]

for i in ls:
   ls1.append(i[0])
   ls2.append(i[1])
   words_initial.append(i[0])
   words_final.append(i[0])
a = len(ls1)

#------------------------------------------------------------------------------
# Word Removal
for i in range(a):
   one = ls1[i]
   two = ls2[i]
   h.write(one)
   h.write(' ')
   h.write(two)
   h.write('\n')
chahiye = ['JJ','JJR','JJS','NN','NNS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
counter=0
index=[]
for tag in ls2:
  if tag not in chahiye:
      index.append(counter)
  counter+=1
for i in index:
   ls1[i]='x'
   ls2[i]='x'
while 'x' in ls1:
   ls1.remove('x')

while 'x' in ls2:
   ls2.remove('x')

#------------------------------------------------------------------------------
# Stemming
from nltk import chain
from nltk import SnowballStemmer
stemmer = SnowballStemmer("english")

#------------------------------------------------------------------------------
# Word Sense Disambiguation
from nltk.corpus import wordnet as wn

def lesk(context_sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
   max_overlaps = 0; lesk_sense = None
   context_sentence = context_sentence.split()
   for ss in wn.synsets(ambiguous_word):
       # If POS is specified.
       if pos and ss.pos is not pos:
           continue

       lesk_dictionary = []

       # Includes definition.
       lesk_dictionary+= wn.synsets(ambiguous_word)[0].definition()
       # Includes lemma_names.
       lesk_dictionary+= ss.lemma_names()

       # Optional: includes lemma_names of hypernyms and hyponyms.
       if hyperhypo == True:
           lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))

       if stem == True: # Matching exact words causes sparsity, so lets match stems.
           lesk_dictionary = [stemmer.stem(i) for i in lesk_dictionary]
           context_sentence = [stemmer.stem(i) for i in context_sentence]

       overlaps = set(lesk_dictionary).intersection(context_sentence)

       if len(overlaps) > max_overlaps:
           lesk_sense = ss
           max_overlaps = len(overlaps)
   return lesk_sense

#------------------------------------------------------------------------------
# Search Function - Index Sequential Search
index = [['a',0],['b',370],['c',574],['d',1015],['e',1343],['f',1584],['g',1767],['h',1905],['i',2016],['j',2303],['k',2331],['l',2348],['m',2468],['n',2686],['o',2748],['p',2852],['q',3259],['r',3291],['s',3539],['t',3951],['u',4114],['v',4186],['w',4186],['x',4262],['y',4263],['z',4269]]

def search(word):
    start=word[0].lower()
    if start.isdigit():
        return 0
    else:
        pos=ord(start)-97
               
    a=index[pos][1]
    b=index[pos+1][1]
    #print(a,b)
    for i in range(a,b):
       dictionary_word=dictionary[i][0]
       if dictionary_word==word:
           return i
       else:
           x = stemmer.stem(str(word))
           y = stemmer.stem(str(dictionary_word))
           #print(i,x,y)
           if x==y:
               return i
               
    return 0

search('ecclesiastic')
#------------------------------------------------------------------------------
# Main body
replacements=[]
for i in range(len(ls1)):
    word=ls1[i]
    found=search(word)
    print(word,found)
    if found:
        answer=lesk(lesk_raw,word)
        if answer is None:
            print(word,"Not found in Lesk")
        else:
            ans=str(answer.name())
            syn=ans.split('.')
            a=syn[0]
            b=syn[1]
            c=int(syn[2])
        
            text =(answer.lemmas()[0].name())
            found2=search(text)
            if found2:
                replace = wn.synsets(text)[0].definition()
                replacements.append([word,replace,b])
            else:
                replacements.append([word,text,b])

for x in replacements:
    temp=x[0]
    pos=words_final.index(temp)
    words_final[pos]=x[1]

original_string=' '.join(words_initial)
lexical_string=' '.join(words_final)
#------------------------------------------------------------------------------

#print(lesk(raw,'inadvertently'))
#print(stemmer.stem("ecclesiastically"))
#print(stemmer.stem("ecclesiastic"))
#print(nltk.pos_tagged())
#print(search('ecclesiastic'))


#import inflect
#p = inflect.engine()
#print(p.a("apple"))

#------------------------------------------------------------------------------
# Phase 2: Fitting the meanings according to Rule Base

def to_singular(word):
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer() 
    singular=lmtzr.lemmatize(word,'n')
    
    return singular

def to_plural(word):
    import inflect
    p = inflect.engine()
    plural=p.plural(word)
    
    return plural

def remove_stopwords(original,meaning_list,word):
    pos=original.index(word)
    pos_tagged=nltk.pos_tag(original)
    prev=pos_tagged[pos-1][1]
    
    meaning_list[0]=meaning_list[0].lower()
    # first word of meaning is an article, so it needs one
    if meaning_list[0] == 'a' or meaning_list[0] == 'an' or meaning_list[0] == 'the':
        # check if prev word is an adj, pronoun or determiner
        if prev=='DT' or prev=='JJ' or prev=='JJR' or prev=='JJS' or prev=='PRP' or prev=='PRP$':
            meaning_list=meaning_list[1:]
        # not an article
        else:
            pass
                           
    # first word of meaning is not an article, so doesn't need one
    else:
        # if prev word is an adj, pronoun or determiner
        if prev=='DT' or prev=='JJ' or prev=='JJR' or prev=='JJS' or prev=='PRP' or prev=='PRP$':
            del original[pos-1]
        # no article before it, nor do we need one
        else:
            pass
            
    meaning=' '.join(meaning_list)
    return meaning

#def add_stopwords():

def is_irregular(word):
    count=0
    for index,row in irregular_verbs.iterrows():
        a=row["present_simple"]
        b=row["past_simple"]
        c=row["past_participle"]
        d=row["present_continous"]
        
        if word==a or word==b or word==c or word==d:
            return index
        if index==138:
            return -1
        count+=1

def get_tense(original,verb,suffix):
    tense='none'
    pos=original.index(verb)
    pos_tagged3=nltk.pos_tag(original)
    
    if suffix=='ing':
        if original[pos-2]=='have' or original[pos-2]=='has' and original[pos-1]=='been':
            tense="present_prefect_continous"
        if original[pos-2]=='had' and original[pos-1]=='been':
            tense="past_perfect_continous"
        if original[pos-3]=='will' and original[pos-2]=='have' and original[pos-1]=='been':
            tense="future_perfect_continous"
        if original[pos-2]=='will' and original[pos-1]=='be':
            tense="future_continous"
        if original[pos-2]=='was' or original[pos-2]=='were':
            tense="past_continous"
        else:
            tense="present_continous"
        
    elif pos_tagged3[pos][1]=='VBN':
        if original[pos-1]=='have' or original[pos-1]=='has':
            tense="present_perfect"
        if original[pos-1]=='had':
            tense="past_perfect"
        if original[pos-1]=='will' and original[pos-1]=='have':
            tense="future_perfect"
        else:
            tense="past_perfect"
    
    elif pos_tagged3[pos][1]=='VB':
        if original[pos-1]=='will':
            tense="simple_future"
        else: 
            tense="simple_present"
            
    elif pos_tagged3[pos][1]=='VBD':
        tense="simple_past"
        
    return tense
    
def get_suffix(word):
    root=stemmer.stem(word)
    for x in range(len(root)):
        if root[x]!=word[x]:
            pos=x
        elif x==len(root)-1:
            pos=x+1
    suffix=word[pos:]
    suffix=''.join(suffix)    
    return suffix

    
def spell_check(word):
    from autocorrect import spell
    word = spell(word)
    return word
    
def replace(original,word,meaning):
    pos=original.index(word)
    original[pos]=meaning
    replaced=' '.join(original)        
    return replaced

def rulebase(original,lexical):
        
    for x in replacements:
        new=""
        word=x[0]
        meaning=x[1]
        part_of_speech=x[2]
        #tense=get_tense(word)
        #suffix=get_suffix(word)
        #root=stemmer.stem(word)
        
        # Adjectives
        if part_of_speech == 's':
            
            #single_phase meaning
            if meaning.find(' ')==-1:
                new=replace(original,word,meaning)
            else:
                pos_tagged=nltk.pos_tag(original)
                adj_pos=original.index(word)
                min_distance=100000
                for i in range(len(original)):
                    if pos_tagged[i][1]=='NN' or pos_tagged[i][1]=='NNS':
                        distance=abs(i-adj_pos)
                        if distance<= min_distance:
                            nearest_noun=pos_tagged[i][0]
                            min_distance=distance
                
                new=replace(original,nearest_noun,meaning)
                new=replace(original,word,nearest_noun)
                
        # Nouns
        elif part_of_speech == 'n':
            
            noun_pos=original.index(word)
            pos_tagged=nltk.pos_tag(original)
            noun_nltk_pos = pos_tagged[noun_pos][1]
            
            #single_phase meaning
            if meaning.find(' ')==-1:      
                if noun_nltk_pos=='NN':
                    noun=to_singular(meaning)
                elif noun_nltk_pos=='NNS':
                    noun=to_plural(meaning)
                
                if original[noun_pos-1] == 'a' or original[noun_pos] == 'an':
                    import inflect
                    p = inflect.engine()
                    prev=p.a(meaning)
                    prev=prev.split(' ')
                    original[noun_pos-1]=prev[0]
                    
                new=replace(original,word,noun)
                
            #phrase_meaning
            else:
                meaning_list=meaning.split(' ')
                meaning=remove_stopwords(original,meaning_list,word)
            
                new=replace(original,word,meaning)
            
        # Adverbs
        elif part_of_speech == 'r':
            adverb=word
            adv_pos=original.index(adverb)
            
            #single_phase meaning
            if meaning.find(' ')==1:
                new=replace(original,adverb,meaning)
            
            #phrase_meaning
            else:
                meaning_list=meaning.split(' ')
                
                answer=lesk(lesk_raw,adverb)
                ans=answer.name()
                syn=ans.split('.')
                
                # Get the simplified meaning
                if syn[0] != adverb:
                    meaning="in a manner which is "+meaning
                
                elif syn[0] == adverb:
                    answer=answer.lemmas()[0].pertainyms()[0].name()
                    correct_answer=lesk(lesk_raw,answer)
                    correct_meaning=correct_answer.lemmas()[0].name()
                    if correct_meaning.find(' ')==1:
                        meaning="in a "+correct_meaning+" manner"
                    else:
                        correct_meaning=correct_answer.definition()
                        meaning="in a manner which is "+correct_meaning
                
                meaning_list=meaning.split(' ')
                print(meaning)
                # Adverb at start
                if adv_pos == 0:
                    meaning[0]=meaning[0].upper()
                    new=replace(original,adverb,meaning)
                    
                # Adverb at end
                elif adv_pos == len(original):
                    new=replace(original,adverb,meaning)
                
                # Anywhere in between
                #else:
                #    new=replace(original,adverb,meaning)
                
                # Anywhere in between
                else:
                    pos_tagged=nltk.pos_tag(original)
                    sub=[]
                    verb='none'
                    obj=[]
                    
                    before=adv_pos-1
                    after=adv_pos+1
                    if pos_tagged[before][1]=='VB' or pos_tagged[before][1]=='VBD' or pos_tagged[before][1]=='VBG' or pos_tagged[before][1]=='VBN' or pos_tagged[before][1]=='VBP' or pos_tagged[before][1]=='VBZ':
                        verb=pos_tagged[before][0]
                        verb_found=-1
                    elif pos_tagged[after][1]=='VB' or pos_tagged[after][1]=='VBD' or pos_tagged[after][1]=='VBG' or pos_tagged[after][1]=='VBN' or pos_tagged[after][1]=='VBP' or pos_tagged[after][1]=='VBZ':
                        verb=pos_tagged[after][0]
                        verb_found=1
                    else:
                        verb='none'
                    
                    # Find the Subject, Object and Predicate
                    # left search
                    sub_found=0
                    for i in range(adv_pos):
                        x=adv_pos-i-1
                        if pos_tagged[x][0]==',' or sub_found==1:
                            break
                        elif pos_tagged[x][1]=='NN' or pos_tagged[x][1]=='NNS' or pos_tagged[x][1]=='NNP' or pos_tagged[x][1]=='NNPS' or pos_tagged[x][1]=='PRP' or pos_tagged[x][1]=='PRP$':
                            sub.append(pos_tagged[x][0])
                            sub_found=1
                    
                    # right search
                    obj_found=0
                    for x in range(adv_pos+1,len(original)):
                        if pos_tagged[x][0]==',' or pos_tagged[x][0]=='.' or obj_found==1:
                            break
                        elif pos_tagged[x][1]=='JJ' or pos_tagged[x][1]=='JJR' or pos_tagged[x][1]=='JJS':
                            obj.append(pos_tagged[x][0])
                        elif pos_tagged[x][1]=='NN' or pos_tagged[x][1]=='NNS' or pos_tagged[x][1]=='NNP' or pos_tagged[x][1]=='NNPS' or pos_tagged[x][1]=='PRP' or pos_tagged[x][1]=='PRP$':
                            obj.append(pos_tagged[x][0])
                            sub_found=1
                    
                    print("Sub= "+' '.join(sub))
                    print("Preicate= "+adverb+" "+verb)
                    print("Object= "+' '.join(obj))
                    temp=[]
                    #case 1: Subject, Object and Predicate all present
                    if len(sub)!=0 and len(obj)!=0 and verb!='none':
                        a=original.index(sub[0])
                        pre=original[:a]
                        b=original.index(obj[-1])+1
                        post=original[b:]
                        
                        temp.append(' '.join(sub))
                        temp.append(verb)
                        temp.append(' '.join(obj))
                        temp.append(meaning)
                        full=pre+temp+post
                        
                    #case 2: only Subject and Predicate found
                    elif len(sub)!=0 and len(obj)==0:
                        a=original.index(sub[0])
                        pre=original[:a]
                        if verb_found==-1:
                            b=adv_pos+1
                        elif verb_found==1:
                            b=original.index(verb)+1
                        post=original[b:]
                        
                        temp.append(' '.join(sub))
                        temp.append(verb)
                        temp.append(meaning)
                        full=pre+temp+post
                    
                    #case 3: only Object and Predicate
                    elif len(sub)==0 and len(obj)!=0:
                        if verb_found==-1:
                            a=original.index(verb)
                        elif verb_found==1:
                            b=adv_pos
                        post=original[b:]    
                        a=original.index(sub[0])
                        pre=original[:a]
                        
                        temp.append(' '.join(obj))
                        temp.append(verb)
                        temp.append(meaning)
                        full=pre+temp+post
                    
                    new=' '.join(full)
                    original=new.split(' ')
                   
        # Verbs
        elif part_of_speech == 'v':
            
            verb=word
            verb_pos=original.index(verb)
            pos_tagged=nltk.pos_tag(original)
            verb_nltk_pos = pos_tagged[verb_pos][1]
            # no suffix for base form(simple present)
            if verb_nltk_pos=='VB':
                verb_suffix=""
            else:
                verb_suffix=get_suffix(verb)
            
            #tense of verb
            verb_tense=get_tense(original,verb,verb_suffix)  
            
            #single_word meaning
            if meaning.find(' ')==-1:
                pos=is_irregular(meaning)
                if pos!=-1:
                    if verb_tense=='present_continous' or verb_tense=='past_continous' or verb_tense=='future_continous' or verb_tense=='present_perfect_continous' or verb_tense=='past_perfect_continous' or verb_tense=='future_perfect_continous':        
                        meaning=irregular_verbs.iloc[pos,3]
                    elif verb_tense=='present_perfect' or verb_tense=='past_perfect' or verb_tense=='future_perfect':
                        meaning=irregular_verbs.iloc[pos,2]
                    elif verb_tense=='simple_present' or verb_tense=='simple_future':
                        meaning=irregular_verbs.iloc[pos,0]
                    elif verb_tense=='simple_past':
                        meaning=irregular_verbs.iloc[pos,1]
                else:
                   root=stemmer.stem(meaning)
                   verb=root+verb_suffix
                   meaning=spell_check(verb) 
                   
                new=replace(original,word,meaning)
            
            #phrase meaning
            else:
                meaning_list=meaning.split(' ')
                pos_tagged2=nltk.pos_tag(meaning_list)  
                
                #check for all verbs in meaning
                for x in range(len(pos_tagged2)):
                    if pos_tagged2[x][1]=='VB' or pos_tagged2[x][1]=='VBD' or pos_tagged2[x][1]=='VBG' or pos_tagged2[x][1]=='VBN' or pos_tagged2[x][1]=='VBP' or pos_tagged2[x][1]=='VBZ':
                        print(pos_tagged2[x][1])
                        verb_pos = is_irregular(pos_tagged2[x][0])
                        
                        # If verb is Irregular
                        if verb_pos!= -1:    
                            if verb_tense=='present_continous' or verb_tense=='past_continous' or verb_tense=='future_continous' or verb_tense=='present_perfect_continous' or verb_tense=='past_perfect_continous' or verb_tense=='future_perfect_continous':        
                                verb=irregular_verbs.iloc[verb_pos,3]
                            elif verb_tense=='present_perfect' or verb_tense=='past_perfect' or verb_tense=='future_perfect':
                                verb=irregular_verbs.iloc[verb_pos,2]
                            elif verb_tense=='simple_present' or verb_tense=='simple_future':
                                verb=irregular_verbs.iloc[verb_pos,0]
                            elif verb_tense=='simple_past':
                                verb=irregular_verbs.iloc[verb_pos,1]
                            
                        # If verb is not irregular
                        elif verb_pos==-1:
                        
                            # Both in same tense
                            if pos_tagged2[x][1]==verb_nltk_pos:
                                verb=pos_tagged2[x][0]
                                
                            else:        
                                root=stemmer.stem(pos_tagged2[x][0])
                                verb=root+verb_suffix
                                verb=spell_check(verb)
                        
                        meaning_list[x]=verb
                
                meaning=' '.join(meaning_list)
                new=replace(original,word,meaning)
                
        else:
            pass                
    
    return new

new_string=rulebase(words_initial,words_final)












