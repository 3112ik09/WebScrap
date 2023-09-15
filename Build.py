import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')


class BuildOutput:
    def __init__(self, test_data):

        ## Return items 
        self.positive_score = None
        self.negative_score = None
        self.polarity_score = None
        self.subjectivity_score = None
        self.avg_sentence_length = None
        self.percentage_of_complex_words = None
        self.fog_index = None
        self.avg_number_of_words_per_sentence = None
        self.complex_word_count = None
        self.word_count = None
        self.syllables_per_word = None
        self.personal_pronouns = None
        self.avg_word_length = None

        # if scraped data is None 
        if test_data ==None:
            return 



        negative_words_path = 'MasterDictionary/negative-words.txt'
        positive_words_path =  'MasterDictionary/positive-words.txt'
        with open(negative_words_path , 'r',encoding='latin-1') as file:
            negative_words = [line.strip() for line in file.readlines()]

        with open(positive_words_path , 'r' ,encoding='latin-1') as file:
            positive_words = [line.strip() for line in file.readlines()]

        # print(positive_words)
        # print(negative_words)

        file_paths = glob.glob("StopWords/"+"*.txt") 
        stop_words_custom=[]
        for file_path in file_paths:
            with open(file_path , 'r' , encoding='latin-1') as file:
                for line in file:
                    words = line.strip().split("|")
                    stop_words_custom.extend(word.strip() for word in words)
      

        
        self.punctuation = set(string.punctuation)
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(stop_words_custom)
        self.test_data = test_data
        self.results = self.sentiment_analysis()


        ## Update the variables  
        self.positive_score = self.results['positive_score']
        self.negative_score = self.results['negative_score']
        self.polarity_score = self.results['polarity_score']
        self.subjectivity_score = self.results['subjectivity_score']
        self.avg_sentence_length = self.find_avg_sentence_length()
        self.percentage_of_complex_words = self.find_percentage_complex_words()
        self.fog_index = self.find_fog_index()
        self.avg_number_of_words_per_sentence = self.find_avg_words_pre_sentence()
        self.complex_word_count = self.find_complex_words()
        self.word_count = self.find_word_count()
        self.syllables_per_word = self.find_syllables_in_text()
        self.personal_pronouns = self.find_personal_pronouns()
        self.avg_word_length = self.find_avg_word_length()

        # Now, add comments for each variable
        # print("Positive Score:", self.positive_score)
        # print("Negative Score:", self.negative_score)
        # print("Polarity Score:", self.polarity_score)
        # print("Subjectivity Score:", self.subjectivity_score)
        # print("Average Sentence Length:", self.avg_sentence_length)
        # print("Percentage of Complex Words:", self.percentage_of_complex_words)
        # print("Fog Index:", self.fog_index)
        # print("Average Number of Words per Sentence:", self.avg_number_of_words_per_sentence)
        # print("Complex Word Count:", self.complex_word_count)
        # print("Word Count:", self.word_count)
        # print("Syllables per Word:", self.syllables_per_word)
        # print("Personal Pronouns:", self.personal_pronouns)
        # print("Average Word Length:", self.avg_word_length)

        # Return items 
      



    def preprocess_text(self, text):
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stop_words and word not in self.punctuation]
        return words

    def sentiment_analysis(self):

        """
        Positive Score: This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.
        Negative Score: This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. We multiply the score with -1 so that the score is a positive number.
        Polarity Score: This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: 
        Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
        Range is from -1 to +1
        Subjectivity Score: This is the score that determines if a given text is objective or subjective. It is calculated by using the formula: 
        Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
        Range is from 0 to +1
        """
        words = self.preprocess_text(self.test_data)

        # Count positive and negative words
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score)+ 0.000001)
        subjectivity_score = (positive_score + negative_score)/ ((len(words)) + 0.000001)

        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'polarity_score': polarity_score,
            'subjectivity_score': subjectivity_score
        }
    
    def find_avg_sentence_length(self):
        """Average Sentence Length = the number of words / the number of sentences"""
        sentences = self.test_data.split('.')
        # print(sentences)
        sentences.pop()
        total_words = 0 
        for sentence in sentences:
            total_words+=len(sentence.split(' '))
        
        average_sentence_length = total_words/len(sentences)
        return average_sentence_length
    
    def find_avg_words_pre_sentence(self ):
        """Average Sentence Length = the number of words / the number of sentences"""
        sentences = self.test_data.split('.')
        sentences.pop()
        total_words = 0 
        for sentence in sentences:
            total_words+=len(sentence.split(' '))
        
        words_pre_sentence = total_words/len(sentences)
        return words_pre_sentence
    
    def find_syllables_per_word(self , word):

        """We count the number of Syllables in each word of the text by counting the vowels present 
        in each word. We also handle some exceptions like words ending with "es","ed" by not counting them as a syllable."""
        if word.endswith('es'):
            word = word[:-2]
        elif word.endswith('ed'):
            word = word[:-2]

        # Count the number of vowels (a, e, i, o, u) in the word
        vowels = 'aeiou'
        syllable_count = sum(1 for letter in word if letter in vowels)

        # Ensure at least one syllable for words with no vowels
        if syllable_count == 0:
            syllable_count = 1

        return syllable_count
    
    def find_syllables_in_text(self): 
        words = self.preprocess_text(self.test_data)
        syllable_counts = sum([self.find_syllables_per_word(word) for word in words])
        return syllable_counts/len(words)
    
    def find_complex_words(self):
        words = word_tokenize(self.test_data)
        count_complex=0
        for word in words:
            no_syllables = self.find_syllables_per_word(word)
            if no_syllables>2:
                count_complex+=1
        return count_complex
    
    def find_percentage_complex_words(self):
        """Percentage of Complex words = the number of complex words / the number of words"""
        words = word_tokenize(self.test_data)
        count_complex=0
        for word in words:
            no_syllables = self.find_syllables_per_word(word)
            if no_syllables>2:
                count_complex+=1
        
        return count_complex/len(words)
    
    def find_personal_pronouns(self):
        words = word_tokenize(self.test_data)  
        personal_pronouns = ["i", "me", "my", "mine", "myself",
                            "you", "your", "yours", "yourself",
                            "he", "him", "his", "himself",
                            "she", "her", "hers", "herself",
                            "it", "its", "itself",
                            "we", "us", "our", "ours", "ourselves",
                            "they", "them", "their", "theirs", "themselves"]
        pronoun_count=0
        for word in words:
            if word in personal_pronouns:
                pronoun_count+=1

        return pronoun_count
    
    def find_fog_index(self):
        """Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)"""
        return 0.4*(self.find_avg_sentence_length()+self.find_percentage_complex_words())
    
    def find_word_count(self):
        return len(self.preprocess_text(self.test_data))
    
    def find_avg_word_length(self):
        """Sum of the total number of characters in each word/Total number of words"""
        words = self.preprocess_text(self.test_data)
        total_length = sum(len(w) for w in words)
        num_words = len(words)
        average_length =0
        if num_words > 0:
            average_length = total_length / num_words
        return average_length


if __name__ == "__main__":
    output_data = []
    df = pd.read_excel('Extract.xlsx')
    print(df.head())
    # print(df.at[0 , 'data'])
    for i in range(0 , len(df)):
        test_data = df.at[i , 'data']
        # print(i)
        # print(test_data[:25])
        if pd.notna(test_data):
            output = BuildOutput(test_data)
        else :
            output = BuildOutput(None)
        data_dict = {
                'Positive Score': output.positive_score,
                'Negative Score': output.negative_score,
                'Polarity Score': output.polarity_score,
                'Subjectivity Score': output.subjectivity_score,
                'Avg Sentence Length': output.avg_sentence_length,
                'Percentage of Complex Words': output.percentage_of_complex_words,
                'Fog Index': output.fog_index,
                'Avg Number of Words per Sentence': output.avg_number_of_words_per_sentence,
                'Complex Word Count': output.complex_word_count,
                'Word Count': output.word_count,
                'Syllables per Word': output.syllables_per_word,
                'Personal Pronouns': output.personal_pronouns,
                'Avg Word Length': output.avg_word_length
        }
        print(data_dict)
        # Append the data dictionary to the list
        output_data.append(data_dict)

    output_df = pd.DataFrame(output_data)
    input_df = pd.read_excel('Input.xlsx')
    merged_df = pd.merge(input_df, output_df, left_index=True, right_index=True, how='inner')

    # Save the merged DataFrame to a new Excel file
    merged_df.to_excel('Output-data-structure.xlsx', index=False)
    # to CSV
    merged_df.to_csv('output.csv', index=False)