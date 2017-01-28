var input_list = ['cat','dog','rats','mouse','hello', 'star', 'man', 'sun', 'god', 'heelo', 'beach','tsar']
;

function anagrams(input) {
    var output = [];

    for(var i = 0; i < input.length; i++){
        for(var j = 0; j < input.length; j++){
            var word1 = input[i];
            var word2 = input[j];

            var w1sort = word1.split('').sort().join();
            var w2sort = word2.split('').sort().join();

            // the equivalent of pdb.set_trace()
            // debugger;

            if((word1 != word2) && (w1sort == w2sort)) {
                output.push(word1);
                break;
            }
        }
    }
    
    return output;
}

console.log(anagrams(input_list));