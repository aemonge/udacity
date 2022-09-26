#!/usr/bin/python3

# Given C -> Cancer, we want to understand the possibilities of having cancer
# Given a Test that indicate a probability of having cancer
ROUND=4

probability_has_cancer = 0.1                            # P(C)
sensitivity_positive = 0.9                              # P(Pos | C)
specificity_positive = 0.8                              # P(Neg | ~C)
#######
probability_not_cancer  = 1 - probability_has_cancer    # P(~C)
sensitivity_negative    = 1 - sensitivity_positive      # P(Neg | C)
specificity_negative    = 1 - specificity_positive      # P(Pos | ~C)

msg=" =========== GIVEN ==========\n\
 C: Cancer\n\
 Pos: Positive Cancer Test\n\
 Neg: Negtive Cancer Test\n\
 === PREMISES WORLD ===\n\
  P(C)        = {}\n\
  P(Pos | C)  = {}\n\
  P(Neg | ~C) = {}\n\
  --- Inverted ---\n\
  P(~C)       = {}\n\
  P(Neg | ~C) = {}\n\
  P(Pos | C)  = {}\n\
"
print(msg.format(
    round(probability_has_cancer, ROUND), round(sensitivity_positive, ROUND), round(specificity_positive, ROUND)
    , round(probability_not_cancer, ROUND), round(sensitivity_negative, ROUND), round(specificity_negative, ROUND)
))

negative_msg="  --- Negatives ---\n\
  P(C, Neg)   = {}\n\
  P(~C, Neg)  = {}\n\
  P(Neg)      = {}\n\
  P(C | Neg)  = {}\n\
  P(~C | Neg) = {}\n\
\n"

probability_with_cancer_negative     = probability_has_cancer * sensitivity_negative
probability_no_cancer_negative       = probability_not_cancer * specificity_negative
probability_negative                 = sum([probability_no_cancer_negative, probability_with_cancer_negative])
probability_given_cancer_negative    = probability_with_cancer_negative / probability_negative
probability_given_no_cancer_negative = probability_no_cancer_negative / probability_negative

print(negative_msg.format(
    round(probability_with_cancer_negative, ROUND), round(probability_no_cancer_negative, ROUND)
    , round(probability_negative, ROUND)
    , round(probability_given_cancer_negative, ROUND), round(probability_given_no_cancer_negative, ROUND)
))

positive_msg="  --- Positives ---\n\
  P(C, Pos)   = {}\n\
  P(~C, Pos)  = {}\n\
  P(Pos)      = {}\n\
  P(~C | Pos) = {}\n\
  P(C | Pos)  = {}\n\
\n"

probability_with_cancer_positive     = probability_has_cancer * sensitivity_positive
probability_no_cancer_positive       = probability_not_cancer * specificity_positive
probability_positive                 = sum([probability_no_cancer_positive, probability_with_cancer_positive])
probability_given_cancer_positive    = probability_with_cancer_positive / probability_positive
probability_given_no_cancer_positive = probability_no_cancer_positive / probability_positive


print(positive_msg.format(
    round(probability_with_cancer_positive, ROUND), round(probability_no_cancer_positive, ROUND)
    , round(probability_positive, ROUND)
    , round(probability_given_cancer_positive, ROUND), round(probability_given_no_cancer_positive, ROUND)
))


# p_C = round(.01 * .9, 10)
# p_nC = round(.99 * .1, 10)
# zum = sum([p_C, p_nC]);

# print("P(C) * P(Pos | C) = {}".format(p_C))
# print("P(~C) * P(Pos | ~C) = {}".format(p_nC))
# print("sum", zum)
# print("P(C | Pos) = {}".format(p_C/zum))
# print("P(~C | Pos) = {}".format(p_nC/zum))
# print("P(~C | Pos) = {}".format(sum([p_nC/zum, p_C/zum])))

# # ==========
# p_nC = 0.99
# sensitivity = 0.1 # P(Pos | C)
# p_posnC = 0.1

# a= 0.01 * 0.1
# b= 0.99 * 0.9
# zum = round(sum([a,b]), 10)
# print("P(C, Neg) = {}".format(round(a, 10)))
# print("P(~C, Neg) = {}".format(round(b, 10)))
# print("== NORMALLY ==", zum)
# print("P(C | Neg) = {}".format(round(a/zum, 10)))
# print("P(~C | Neg) = {}".format(round(b/zum, 10)))
