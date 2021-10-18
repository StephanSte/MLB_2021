valid_mails = [
    "Dear Arthur I have news for you, my neace has recently married a thief in Valentine",
    "Oh Arthur I fear for her life please look after her",
    "I wish I would have run away with you when I had the chance but you know daddy no one can look after him but me",
    "I I I the egg is the Ei",
    "Brother Brother Brother Brother Brother Brother Brother Brother Brother Brother"
]

spam_mails = [
    "Friend you win mucho dinero money is all you get",
    "Sell me your soul and i will make you rich as Jeff bezos i work at Google and make lot money yes Friend",
    "Friend i have a business idea, we will make money i tell you its not a pyramid scheme i swear",
    "money money money money money money money money money money money money money money"
]


class MailClass:
    spam_prop = 1
    valid_prop = 1
    wordCounts_spam = {}
    wordCounts_valid = {}
    wordProbabilities_spam = {}
    wordProbabilities_valid = {}
    spam_count = 0
    valid_count = 0

    def __init__(self, spam, valid):
        self.spam_prop = len(spam) / (len(spam) + len(valid))
        self.non_spam_prop = len(valid) / (len(spam) + len(valid))

        self.wordCounts_spam = {}
        for mail in spam:
            for word in mail.split():
                self.spam_count += 1
                if word not in self.wordCounts_spam:
                    self.wordCounts_spam[word] = 1
                self.wordCounts_spam[word] += 1

        self.wordCounts_valid = {}
        for mail in valid:
            for word in mail.split():
                self.valid_count += 1

                if word not in self.wordCounts_valid:
                    self.wordCounts_valid[word] = 1
                self.wordCounts_valid[word] += 1

        self.wordProbabilities_spam = {}
        for word in self.wordCounts_spam:
            self.wordProbabilities_spam[word] = int(self.wordCounts_spam[word]) / self.valid_count

        self.wordProbabilities_valid = {}
        for word in self.wordCounts_valid:
            self.wordProbabilities_valid[word] = int(self.wordCounts_valid[word]) / self.spam_count

    def predict(self, mail):
        for word in mail.split():
            if word not in self.wordProbabilities_valid:
                self.valid_count += 1
                temp_prob_valid = 1 / self.valid_count
                self.valid_prop *= temp_prob_valid
                continue
            self.valid_prop *= self.wordProbabilities_valid[word]

        for word in mail.split():
            if word not in self.wordProbabilities_spam:
                self.spam_count += 1
                temp_prob_spam = 1 / self.spam_count
                self.spam_prop *= temp_prob_spam
                continue
            self.spam_prop *= self.wordProbabilities_spam[word]

        print(self.spam_prop)

        print(self.valid_prop)

        return "spam" if self.spam_prop > self.valid_prop else "not spam"


pred = MailClass(spam_mails, valid_mails)

test_mail = "Brother Arthur I"

print(pred.predict(test_mail))
