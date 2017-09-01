//============================================================================
// Name        : LogisticRegression.cpp
// Author      :
// Version     :
// Copyright   :
// Description : Implementation of logistic regression using homomorphic encryption
//============================================================================

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <cstdlib>
#include <cstdio>
#include </usr/include/mlpack/core.hpp>
#include </usr/include/mlpack/methods/linear_regression/linear_regression.hpp>
#include "seal.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "encryptor.h"
#include "encryptionparams.h"
#include "keygenerator.h"
#include "evaluationkeys.h"
#include "bigpoly.h"
#include "biguint.h"
#include "bigpolyarray.h"
#include "chooser.h"
#include "evaluator.h"

using namespace std;

class LogisticRegression {
public:
	LogisticRegression() {

	}

	LogisticRegression(seal::Ciphertext** theta, seal::Ciphertext** data,
			seal::Ciphertext* target) {
		this->theta = theta;
		this->data = data;
		this->target = target;
	}

	// class 1 if log(h(x)) >= log(thresh) else class 0
	seal::Ciphertext* predict(int col, int row, seal::Ciphertext** data,
			seal::Ciphertext** theta, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Evaluator evaluate) {
		seal::Ciphertext *res = new seal::Ciphertext[col];
		seal::Ciphertext *pred = new seal::Ciphertext[col];

		seal::Ciphertext add;
		for (int j = 0; j < col; j++) {
			pred[j] = log_h(false, data[j], theta, a0, a1, a2, evaluate, row,
					col);
		}
		return pred;
	}

	seal::Ciphertext** train(int iters, int col, int row,
			seal::Ciphertext** data, seal::Ciphertext** theta,
			seal::Ciphertext* target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha) {
//		for (int i = 0; i <= row; i++) {
//			cout << theta[i][0].operator seal::BigPolyArray &().coeff_count();
//			cout
//					<< theta[i][0].operator seal::BigPolyArray &().coeff_bit_count();
//
//		}

		seal::Ciphertext **t = new seal::Ciphertext*[1];
		cout << "in train" << endl;
//		seal::Ciphertext **th = new seal::Ciphertext*[1];
//		for (int i = 0; i <= row; i++) {
//			cout << i << endl;
//			th[i] = new seal::Ciphertext[1];
//			seal::Ciphertext tht = theta[i][0];
//			cout << "train2" << endl;
//			th[i][0] = tht;
//			cout << "train3" << endl;
//		}

		seal::Ciphertext **thet = gradient_descent(iters, col, row, data, theta,
				target, evaluate, fr, a0, a1, a2, minus, tr, alpha);
		cout << "train works fine" << endl;
		t = thet;
		return t;
	}

private:
	seal::Ciphertext** theta;
	seal::Ciphertext** data;
	seal::Ciphertext* target;

	//theta[j] = theta[j] + alpha (y[i] - log_h(x[i])* x[i][j]
	seal::Ciphertext** gradient_descent(int iterations, int col, int row,
			seal::Ciphertext** data, seal::Ciphertext** theta,
			seal::Ciphertext* target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha) {
		cout << "in gradient descent" << endl;
		//	seal::Ciphertext **t_tmp = new seal::Ciphertext*[1];
		//	cout << row << endl;
		seal::Ciphertext **thet = new seal::Ciphertext*[1];
//		for (int i = 0; i <= row; i++) {
//			cout << i << endl;
//			t_tmp[i] = new seal::Ciphertext[1];
//			cout << "worked" << endl;
//			seal::Ciphertext t = theta[i][0];
//			cout << "fine" << endl;
//			t_tmp[i][0] = t;
//		}
		seal::Ciphertext* J = new seal::Ciphertext[iterations];
		for (int i = 0; i < iterations; i++) {
			seal::Ciphertext *pred = new seal::Ciphertext[col];
			seal::Ciphertext *diff = new seal::Ciphertext[col];

			for (int j = 0; j < col; j++) {
				//		cout << j << endl;
				pred[j] = log_h(false, data[j], theta, a0, a1, a2, evaluate,
						row, col);

				diff[j] = seal::Ciphertext(
						evaluate.sub(
								target[j].operator const seal::BigPolyArray &(),
								pred[j].operator const seal::BigPolyArray &()));
				//	cout << "sub fine" << endl;
			}
			seal::Ciphertext *res = new seal::Ciphertext[row + 1];
			seal::Ciphertext *r = new seal::Ciphertext[row + 1];

			//	cout << "sub total" << endl;
			for (int k = 0; k <= row; k++) {
				//	cout << "row " << k << endl;
				thet[k] = new seal::Ciphertext[1];
				for (int j = 0; j < col; j++) {
					//		cout << "col " << j << endl;
					res[k] =
							seal::Ciphertext(
									evaluate.relinearize(
											seal::Ciphertext(
													evaluate.multiply(
															diff[k].operator const seal::BigPolyArray &(),
															data[j][k].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
					//	cout << "multiply fine" << endl;
					r[k] =
							seal::Ciphertext(
									evaluate.relinearize(
											seal::Ciphertext(
													evaluate.multiply_plain(
															res[k].operator const seal::BigPolyArray &(),
															alpha)).operator const seal::BigPolyArray &()));
					//	cout << "multiply plain fine" << endl;

					thet[k][0] =
							seal::Ciphertext(
									evaluate.add(
											theta[k][0].operator const seal::BigPolyArray &(),
											r[k].operator const seal::BigPolyArray &()));
					//	cout << "add fine" << endl;

				}
			}
			cout << "before compute in grAD" << endl;
			J[i] = compute_cost(col, row, thet, data, target, fr, evaluate, a0,
					a1, a2, minus, tr);
			cout << "fine compute cost" << endl;
		}
		return thet;
	}

//computes:	a0 -a1*(theta* x) + a2*(theta* x)² for a single data point;
//if neg=true compute log(1-h(x))
	seal::Ciphertext log_h(bool neg, seal::Ciphertext* data,
			seal::Ciphertext** theta, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Evaluator evaluate, int row, int col) {
		seal::Ciphertext res;
		vector<seal::Ciphertext> ve;
//		cout << "in log_h" << endl;
		for (int i = 0; i <= row; i++) {
			// theta * x
			//	cout << i << endl;
			seal::Ciphertext t =
					seal::Ciphertext(
							evaluate.relinearize(
									seal::Ciphertext(
											evaluate.multiply(
													data[i].operator const seal::BigPolyArray &(),
													theta[i][0].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
			ve.emplace_back(t);
			//	cout << "worked" << endl;
		}
		//cout << "theta*x works" << endl;
		seal::Ciphertext tmp = evaluate.add_many(ve);
		// term1 = (theta*x) * a1
		seal::Ciphertext term1 = seal::Ciphertext(
				evaluate.multiply_plain(
						tmp.operator const seal::BigPolyArray &(), a1));
		//(theta*x)²
		seal::Ciphertext sq_tmp = seal::Ciphertext(
				evaluate.relinearize(
						evaluate.square(
								tmp.operator const seal::BigPolyArray &())));
		//term2 = (theta*x)² * a2
		seal::Ciphertext term2 = seal::Ciphertext(
				evaluate.relinearize(
						evaluate.multiply_plain(
								sq_tmp.operator const seal::BigPolyArray &(),
								a2)));
		// a0 + term 2
		seal::Ciphertext add = seal::Ciphertext(
				evaluate.add_plain(term2.operator const seal::BigPolyArray &(),
						a0));
		// log(1- h(x))
		if (neg) {
			seal::Ciphertext sub = seal::Ciphertext(
					evaluate.add(add.operator const seal::BigPolyArray &(),
							term1.operator const seal::BigPolyArray &()));
			res = sub;
		}
		// log(h(x))
		else {
			// a0- term1 + term2
			seal::Ciphertext sub = seal::Ciphertext(
					evaluate.sub(add.operator const seal::BigPolyArray &(),
							term1.operator const seal::BigPolyArray &()));
			res = sub;
		}
		//	cout << "log_h works" << endl;
		return res;
	}
	/*
	 seal::Ciphertext compute_A(int j, int col, seal::Ciphertext* targets,
	 seal::Ciphertext** data, seal::Ciphertext** theta,
	 seal::Evaluator evaluate, seal::FractionalEncoder encoder,
	 seal::Plaintext minus_one) {
	 seal::Ciphertext res;
	 seal::Ciphertext add;
	 seal::Ciphertext* dat = new seal::Ciphertext[j];
	 if (j % 2 == 0) {

	 for (int i = 0; i < col; i++) {
	 for (int k = 0; k < j; k++) {
	 dat[k] = data[i][k];
	 }
	 seal::Ciphertext tx = seal::Ciphertext(
	 evaluate.multiply_plain(
	 dat[i].operator const seal::BigPolyArray &(),
	 minus_one.operator const seal::BigPoly &()));

	 }
	 } else {
	 for (int i = 0; i < col; i++) {
	 for (int k = 0; k < j; k++) {
	 dat[k] = data[i][k];
	 }
	 add =
	 seal::Ciphertext(
	 evaluate.add(
	 targets->operator const seal::BigPolyArray &(),
	 targets->operator const seal::BigPolyArray &()));
	 add = seal::Ciphertext(
	 evaluate.sub_plain(
	 add.operator const seal::BigPolyArray &(),
	 minus_one.operator const seal::BigPoly &()));

	 seal::Ciphertext tx = seal::Ciphertext(
	 evaluate.multiply(
	 dat[i].operator const seal::BigPolyArray &(),
	 add.operator const seal::BigPolyArray &()));
	 }
	 }
	 return res;
	 } */

//compute J'(theta)
	seal::Ciphertext compute_helper(int col, int row, seal::Ciphertext** theta,
			seal::Ciphertext** data, seal::Ciphertext* target,
			seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
			seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus_one) {
		vector<seal::Ciphertext> v;
		seal::Ciphertext res;
	//	cout << "in compuite helper" << endl;
		for (int i = 1; i < col; i++) {
			// -y(i)
	//		cout << i << endl;
			seal::Ciphertext tmp1 = seal::Ciphertext(
					evaluate.negate(
							target[i].operator const seal::BigPolyArray &()));
			// -y(i)*log_h
			seal::Ciphertext t1 =
					seal::Ciphertext(
							evaluate.relinearize(
									seal::Ciphertext(
											evaluate.multiply(
													tmp1.operator const seal::BigPolyArray &(),
													log_h(false, data[i], theta,
															a0, a1, a2,
															evaluate, row,
															col))).operator const seal::BigPolyArray &()));
			//(y(i)-1)
			seal::Ciphertext tmp2 = seal::Ciphertext(
					evaluate.sub_plain(
							target[i].operator const seal::BigPolyArray &(),
							minus_one));
			//(y(i)-1)* log_h
			seal::Ciphertext t2 =
					seal::Ciphertext(
							evaluate.relinearize(
									seal::Ciphertext(
											evaluate.multiply(
													tmp2.operator const seal::BigPolyArray &(),
													log_h(true, data[i], theta,
															a0, a1, a2,
															evaluate, row,
															col))).operator const seal::BigPolyArray &()));

			// t1+ t2
			seal::Ciphertext tre = seal::Ciphertext(
					evaluate.add(t1.operator const seal::BigPolyArray &(),
							t2.operator const seal::BigPolyArray &()));
			v.emplace_back(tre);
		}
		// summing up J'(theta) for each data point
		seal::Ciphertext re = evaluate.add_many(v);
		// multiply with fr=1/col
		res = seal::Ciphertext(
				evaluate.multiply_plain(
						re.operator const seal::BigPolyArray &(),
						fr.operator const seal::BigPoly &()));
		return res;
	}
	// fr = 1/col
	//Plaintextm minus_one = -1 encoded
	// J(theta) = lambda/(2col) * sum(i=1,d) theta[i]² + J'(theta) with J'(theta) = compute_helper
	seal::Ciphertext compute_cost(int col, int row, seal::Ciphertext** theta,
			seal::Ciphertext** data, seal::Ciphertext* target,
			seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
			seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus_one,
			seal::Plaintext tr) {
		seal::Ciphertext sum;
		seal::Ciphertext res;
		vector<seal::Ciphertext> ve;
	//	cout << "in compute_cost" << endl;
		for (int i = 1; i <= row; i++) {
			seal::Ciphertext sq = seal::Ciphertext(
					evaluate.square(
							theta[i][0].operator const seal::BigPolyArray &()));
			//	cout << i << endl;
			ve.emplace_back(sq);

		}
//		cout << "sum fine" << endl;
		sum = evaluate.add_many(ve);
		seal::Ciphertext tl = compute_helper(col, row, theta, data, target, fr,
				evaluate, a0, a1, a2, minus_one);
	//	cout << "after compujte helper" << endl;
		seal::Ciphertext tl2 = seal::Ciphertext(
				evaluate.add(tl.operator const seal::BigPolyArray &(),
						sum.operator const seal::BigPolyArray &()));
	//	cout << "after add" << endl;
		res = seal::Ciphertext(
				evaluate.multiply_plain(
						tl2.operator const seal::BigPolyArray &(),
						tr.operator const seal::BigPoly &()));
	//	cout << "after multiply plain" << endl;
		return res;
	}
};
