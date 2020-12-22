module SP_reader


"""
helper module, with functions to read and pre-process data
"""


users = r"@[^ ]+"
numbers = r"[0-9]"
urls = r"(https?:\/\/)?(?:www\.|(?!www))?[^\s\.]+\.[^\s]{2,}|(www)?\.[^\s]+\.[^\s]{2,}"


function read_conll_file(file::AbstractString)

    """
    reads in a ´file´ in CoNLL format:
    ´´´
        word1    tag1
        ...      ...
        wordN    tagN
    ´´´
    N.B: sentences are assumed to be already tokenized!
    N.B.2: there should be a blank line to separate each sentence
    ´´´
    Retrieves one instance for each sentence, where an instance is a tuple of arrays (words[],tags[])
    Instances are stored and returned in a channel.
    """

    Channel() do channel

        current_words = AbstractString[]
        current_tags = AbstractString[]

        open(file,"r") do io
            while !eof(io)
                line = strip(readline(io))

                if !isempty(line)
                    word, tag = split(line, "\t")
                    push!(current_words,word)
                    push!(current_tags,tag) 
                else
                    put!(channel, (current_words,current_tags)) 
                    current_words = AbstractString[]
                    current_tags = AbstractString[]
                end
            end
        end
    end
end


function normalize!(word::AbstractString)
    
    """
    basic text pre-processing: lowercase ´word´, and replace numbers, user names, and URLs 
    ´´´
    substitutions are carried out inplace
    """

    return replace((replace(replace(lowercase(word), numbers => "0"), users => "@USER")), urls => "URL")
end


end

