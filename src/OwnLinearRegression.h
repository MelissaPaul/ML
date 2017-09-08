
#ifndef OWNLINEARREGRESSION_H_
#define OWNLINEARREGRESSION_H_


#include "seal.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "encryptor.h"
#include "evaluator.h"

class OwnLinearRegression {
public:

// trains a linear regression model
/**
 *
 * @param alpha: learning rate as double
 * @param iterations: # of iterations for gradient descent
 * @param n_col: # of columns
 * @param x: encrypted data (two dimensional array)
 * @param y: encrypted targets (one dimensional array)
 * @param th: encrypted weights (two dimensional array)
 * @param n_row: # of rows
 * @param text: Plaintext
 * @param evaluate: seal::Evaluator
 * @param ridge: boolean; true of ridge regression to be performed
 * @param lambda_div: double lambda divided by n_col
 * @return
 */
    seal::Ciphertext **train(seal::Plaintext alpha, int iterations, int n_col,
                             seal::Ciphertext **x, seal::Ciphertext y[], seal::Ciphertext **th,
                             int n_row, seal::Plaintext text, seal::Evaluator evaluate,
                             bool ridge, seal::Plaintext lambda_div);
// predicts the outcome of a data set
/**
 *
 * @param n_row: # of rows
 * @param n_col: # of columns
 * @param x: encrypted data (two dimensional array)
 * @param theta: encrypted weights (two dimensional array)
 * @param evaluate: seal::Evaluator
 * @return
 */
    seal::Ciphertext *predict(int n_row, int n_col, seal::Ciphertext **x,
                              seal::Ciphertext **theta, seal::Evaluator evaluate);

private:
    // computes cost function J
    /**
     *
     * @param n_row
     * @param n_col
     * @param x: encrypted data (two dimensional array)
     * @param y: encrypted targets (one dimensional array)
     * @param theta: encrypted weights (two dimensional array)
     * @param evaluate: seal::Evaluator
     * @param con: (1.0 / (2 * n_col)) encoded as seal::Plaintext
     * @param ridge: boolean; true of ridge regression to be performed
     * @param lambda_div: double lambda divided by n_col
     * @return
     */
    seal::Ciphertext compute_cost(int n_row, int n_col, seal::Ciphertext **x,
                                  seal::Ciphertext y[], seal::Ciphertext **theta,
                                  seal::Evaluator evaluate, seal::Plaintext con, bool ridge,
                                  seal::Plaintext lambda_div);

    //makes acutal prediction (help function)
    /**
     *
     * @param x: encrypted data (two dimensional array)
     * @param theta: encrypted weights (two dimensional array)
     * @param n_col: # of columns
     * @param evaluate: seal::Evaluator
     * @return
     */
    seal::Ciphertext h(seal::Ciphertext x[], seal::Ciphertext **theta,
                       int n_col, seal::Evaluator evaluate);

/**
 *
 * @param n_row : # of columns
 * @param n_col: # of columns
 * @param x: encrypted data (two dimensional array)
 * @param theta: encrypted weights (two dimensional array)
 * @param evaluate: seal::Evaluator
 * @return
 */
    seal::Ciphertext *calculate_predictions(int n_row, int n_col,
                                            seal::Ciphertext **x, seal::Ciphertext **theta,
                                            seal::Evaluator evaluate);

    //performs gradient descent
    /**
     *
     * @param n_col: # of columns
     * @param theta: encrypted weights (two dimensional array)
     * @param x: encrypted data (two dimensional array)
     * @param y: encrypted targets (one dimensional array)
     * @param alpha: learning rate as double
     * @param iters: # of iterations for gradient descent
     * @param J: stores result of cost function (one dimensional array)
     * @param evaluate: seal::Evaluator
     * @param n_row: # of columns
     * @param text:((1.0 / (2.0 * n_col)) encoded as seal::Plaintext
     * @param ridge: boolean; true of ridge regression to be performed
     * @param lambda_div: double lambda divided by n_col
     * @return
     */
    seal::Ciphertext **gradient_descent(int n_col, seal::Ciphertext **theta,
                                        seal::Ciphertext **x, seal::Ciphertext y[], seal::Plaintext alpha,
                                        int iters, seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
                                        seal::Plaintext text, bool ridge, seal::Plaintext lambda_div);
};

#endif /* OWNLINEARREGRESSION_H_ */
