//============================================================================
// Name        : bt.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
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
using namespace seal;
using namespace mlpack;
using namespace mlpack::regression;
//performs gradient descent
seal::Ciphertext ** gradient_descent(int n_col, seal::Ciphertext **theta,
		seal::Ciphertext **x, seal::Ciphertext y[], Plaintext alpha, int iters,
		seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
		Plaintext text);
// trains a linear regression model
Ciphertext** train(Plaintext alpha, int iterations, int n_col, Ciphertext **x,
		Ciphertext y[], Ciphertext **th, int n_row, Plaintext text,
		seal::Evaluator evaluate);
// predicts the outcome of a data set
Ciphertext* predict(int n_row, int n_col, seal::Ciphertext **x,
		Ciphertext **theta, seal::Evaluator evaluate);
// compuites cost function J
seal::Ciphertext compute_cost(int n_row, int n_col, seal::Ciphertext **x,
		seal::Ciphertext y[], seal::Ciphertext **theta,
		seal::Evaluator evaluate, Plaintext con);
//makes acutal prediction (help function)
seal::Ciphertext h(seal::Ciphertext x[], seal::Ciphertext **theta, int n_col,
		seal::Evaluator evaluate);
seal::Ciphertext * calculate_predictions(int n_row, int n_col,
		seal::Ciphertext **x, seal::Ciphertext **theta,
		seal::Evaluator evaluate);

////set parameters
//EncryptionParameters setparms() {
//	EncryptionParameters parms;
//	parms.set_poly_modulus("1x^2048+1");
//	parms.set_coeff_modulus(
//			ChooserEvaluator::default_parameter_options().at(2048));
//	parms.set_plain_modulus(1 << 8);
//	parms.validate();
//	return parms;
//}

//Ciphertext encrypt(string& input, EncryptionParameters parms) {

// String to int
//	string::size_type sz;
//	double b = stod(input, &sz);
//cout << "string as int" << b << endl;

// encode text as plaintext in polynomials
//FractionalEncoder encoder(parms.plain_modulus());
//	Plaintext encoded = encoder.encode(b);
//cout << "Encoded number" << encoded.to_string() << endl;

//Generate Keys
//KeyGenerator generator(parms);
//generator.generate();
//const BigPolyArray p_key = generator.public_key();
//Ciphertext public_key = Ciphertext(p_key);
//const BigPoly s_key = generator.secret_key();
//Plaintext secret_key = Plaintext(s_key);
// encrypt text
//Encryptor encryptor(parms, public_key);
//Ciphertext encrypted = encryptor.encrypt(encoded);
//return encrypted;
//}

//encrypts an integer by turning it into Plaintext (polynomial) and then Ciphertext
Ciphertext encrypt(const int input, Encryptor encryptor,
		IntegerEncoder encoder) {
	Plaintext encoded = encoder.encode(input);
	Ciphertext encrypted = encryptor.encrypt(encoded);
	return encrypted;
}
Ciphertext encrypt_frac(double input, Encryptor encryptor,
		FractionalEncoder encoder) {
	Plaintext encoded = encoder.encode(input);
	Ciphertext encrypted = encryptor.encrypt(encoded);
	return encrypted;
}

// decrypts given ciphertext by turning it into Plaintext and then int
int decrypt(Ciphertext encrypted, Decryptor decryptor, IntegerEncoder encoder) {
	cout << "decryptor" << endl;
	BigPoly decrypted = decryptor.decrypt(
			encrypted.operator const seal::BigPolyArray &());
	int decryptt = encoder.decode_int32(decrypted);
	cout << "int" << endl;
	return decryptt;
}

// decrypt given Ciphertext to double
double decrypt_frac(Ciphertext encrypted, Decryptor decryptor,
		FractionalEncoder encoder) {
	cout << "decryptor" << endl;
	BigPoly decrypted = decryptor.decrypt(
			encrypted.operator const seal::BigPolyArray &());
//	cout << "Plaintext" << endl;
	double decryptt = encoder.decode(decrypted);
	cout << "int" << endl;
	return decryptt;
	// Decode results.
}

////perform linear regression

arma::vec reg_lin_reg(arma::mat data, arma::vec responses, double lambda) {

//	// perform linear regression; lambda = 0.5 for ridge regression; 0.0 for linear regression
	LinearRegression lr(data, responses, lambda);
	arma::vec parameters = lr.Parameters();
//
	arma::mat train_data = data.submat(0, 0, 3, 11);
	arma::vec train_target = responses.subvec(0, 11);
	arma::mat test_data = data.submat(0, 11, 3, 22);
	arma::vec test_target = responses.subvec(11, 22);
//	lr.Train(train_data, train_target, false, );
	arma::vec predictions;
	lr.Predict(data, predictions);
	return predictions;
}

//Ciphertext* reg_lin_reg_enc(Ciphertext[][23] encoded, double lambda) {
//	//	// perform linear regression; lambda = 0.5 for ridge regression; 0.0 for linear regression
//	LinearRegression lr(data, responses, lambda);
//	arma::vec parameters = lr.Parameters();
//	//
//	arma::mat train_data = data.submat(0, 0, 3, 11);
//	arma::vec train_target = responses.subvec(0, 11);
//	arma::mat test_data = data.submat(0, 11, 3, 22);
//	arma::vec test_target = responses.subvec(11, 22);
//	//	lr.Train(train_data, train_target, false, );
//	arma::vec predictions;
//	lr.Predict(data, predictions);
//	return predictions;
//}

/*int* string2ascii(string input) {
 int* ptr = new int[input.length() + 1];
 for (int i = 0; i < input.length(); i++) {
 ptr[i] = (int) input[i];
 }
 ptr[input.length()] = -1;
 return ptr;
 }

 string ascii2string(int *ptr) {
 int length = sizeof(ptr) / sizeof(ptr[0]);
 stringstream ss;
 while (*(ptr) != -1) {
 ss << (char) *(ptr);
 cout << (char) *(ptr) << endl;
 ptr++;
 }
 return ss.str();
 }*/
//LinearRegression::LinearRegression(seal::Ciphertext x[][],
//		seal::Ciphertext y[], int n_col) {
//	this->x = x;
//	this->y = y;
//	this->n_col = c_col;
//}
/*Ciphertext* get_column(Ciphertext** x, int column, int n_row, int n_col) {
 Ciphertext res[n_col];
 for (int i = 0; i <= n_row; i++) {
 res[i] = x[i][column];
 cout << i << endl;
 }
 cout << "worked fine" << endl;
 return res;
 }*/

// train the linear regression model, i.e compute the coefficients theta
seal::Ciphertext** train(seal::Plaintext alpha, int iterations, int n_col,
		seal::Ciphertext **x, seal::Ciphertext y[], seal::Ciphertext **th,
		int n_row, seal::Plaintext text, seal::Evaluator evaluate) {
	seal::Ciphertext J[iterations];
	//seal::Ciphertext** theta= new seal::Ciphertext*[1];
	cout << "init all right" << endl;
	seal::Ciphertext **theta = new seal::Ciphertext*[1];
	seal::Ciphertext** tht = new seal::Ciphertext*[1];
	for (int i = 0; i <= n_row; i++) {
		tht[i] = new seal::Ciphertext[1];
		//	theta[i] = new seal::Ciphertext[1];
		tht[i][0] = th[i][0];
	}
	seal::Ciphertext **thet = gradient_descent(n_col, tht, x, y, alpha,
			iterations, J, evaluate, n_row, text);
	cout << "train works fine" << endl;
	theta = thet;
	return theta;
}

// predict the target for each data
seal::Ciphertext* predict(int n_row, int n_col, seal::Ciphertext **x,
		seal::Ciphertext **theta, seal::Evaluator evaluate) {
	seal::Ciphertext *res = new seal::Ciphertext[n_col];
	//fill_n(res, n_col, new seal::Ciphertext());
//	for (int i = 0; i < n_col; i++) {
//		res[i] = new seal::Ciphertext();
//	}
//	seal::Ciphertext tmp = h(*x, theta, n_row, evaluate);
//	cout << "start predict" << endl;
//	res[0] = tmp;
	seal::Ciphertext** tht = new seal::Ciphertext*[1];
	for (int i = 0; i <= n_row; i++) {
		tht[i] = new seal::Ciphertext[1];
		//	theta[i] = new seal::Ciphertext[1];
		tht[i][0] = theta[i][0];
	}
	for (int i = 0; i < n_col; i++) {
		//cout << i << endl;
		Ciphertext t = h(*(x + i), tht, n_row, evaluate);
		//	cout << "h works" << endl;
		res[i] = t;
		//	cout << "res works" << endl;
		//res++;
//		cout << res[i].operator seal::BigPolyArray &().coeff_count()
//				<< " predict "
//				<< res[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;

	}
	seal::Ciphertext *r = new seal::Ciphertext[n_col];
	r = res;
	cout << "prediction works fine" << endl;
	return r;
}

// compute the cost function J(theta)
seal::Ciphertext compute_cost(int n_row, int n_col, seal::Ciphertext **x,
		seal::Ciphertext y[], seal::Ciphertext **theta,
		seal::Evaluator evaluate, Plaintext con) {
	//calculate predictions

	seal::Ciphertext *predictions = calculate_predictions(n_row, n_col, x,
			theta, evaluate);

	// stores difference h(x) -y for each data point
	seal::Ciphertext diff[n_col];
	//stores square of diff for each data point
	seal::Ciphertext sq_errors[n_col];
	//compute (h(x)-y)^2
	for (int i = 0; i < n_col; i++) {
		//	cout << i << endl;
		diff[i] = seal::Ciphertext(
				evaluate.sub(
						predictions[i].operator const seal::BigPolyArray &(),
						y[i].operator const seal::BigPolyArray &()));
		//	cout << "after diff" << endl;
		sq_errors[i] =
				seal::Ciphertext(
						evaluate.relinearize(
								seal::Ciphertext(
										evaluate.square(
												diff[i].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
		//	cout << "sq" << sq_errors[i].size() << endl;
		//cout << "after sq" << endl;
	}
	// creates sq_errors as a vector to add all entries of the data at once
	vector<seal::Ciphertext> t;
	for (int i = 0; i < n_col; i++) {
		t.emplace_back(sq_errors[i]);
	}

//	cout << "vector done" << endl;
	//add all
	seal::Ciphertext r = evaluate.add_many(t);

	//cout << "add finished" << endl;
	//multiply result by (1.0 / (2 * n_col))
	seal::Ciphertext res = Ciphertext(
			evaluate.multiply_plain(r.operator const seal::BigPolyArray &(),
					con));
	//cout << "multiply done" << endl;
	return res;
}

//helper function that does actual prediction
seal::Ciphertext h(seal::Ciphertext x[], seal::Ciphertext **theta, int n_row,
		seal::Evaluator evaluate) {
	seal::Ciphertext* res = new seal::Ciphertext[n_row + 1];
	cout << "inside h " << endl;
	/*
	 seal::Ciphertext tmp = seal::Ciphertext(
	 evaluate.multiply((**theta).operator const seal::BigPolyArray &(),
	 x[0].operator const seal::BigPolyArray &()));
	 */
//	cout << "0 before" << (**theta).operator seal::BigPolyArray &().size()
//			<< " and " << x[0].operator seal::BigPolyArray &().size() << endl;
	//cout << x[0].operator seal::BigPolyArray &().size() << endl;
	seal::Ciphertext tmp =
			seal::Ciphertext(
					evaluate.relinearize(
							seal::Ciphertext(
									evaluate.multiply(
											(**theta).operator const seal::BigPolyArray &(),
											x[0].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
	//cout << "tmp works" << endl;
	//cout << "after" << tmp.size() << endl;
	*res = tmp;
//	cout << "first multiplz worked " << endl;
	// multiply theta with corresponding x
	for (int i = 1; i <= n_row; i++) {
		//	cout << i << endl;
		//	cout << theta[i][0].operator seal::BigPolyArray &().coeff_count()
		//			<< "  "
		//			<< theta[i][0].operator seal::BigPolyArray &().coeff_bit_count()
		//			<< endl;
		seal::Ciphertext tmo = seal::Ciphertext(
				evaluate.multiply(
						theta[i][0].operator const seal::BigPolyArray &(),
						x[i].operator const seal::BigPolyArray &()));
		//	cout << "multiply fine" << endl;
		*(res + i) = seal::Ciphertext(
				evaluate.relinearize(
						tmo.operator const seal::BigPolyArray &()));
//		cout << res[i].operator seal::BigPolyArray &().coeff_count() << " in h "
//				<< res[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;
//
		/*seal::Ciphertext(
		 evaluate.multiply(
		 theta[i][0].operator const seal::BigPolyArray &(),
		 x[i].operator const seal::BigPolyArray &()));*/
//		seal::Ciphertext tmp = *(res + i);
		//	cout << tmp.size() << endl;
		//		cout << "multiplyinhg" << endl;
	}
//create vector of result of the loop theta_0 * x_0, ..., theta_n * x_n
	vector<seal::Ciphertext> t;
//	cout << "vector created" << endl;
	for (int i = 0; i <= n_row; i++) {
		t.emplace_back(res[i]);
	}
//	cout << "Vector filled" << endl;
// add all ciphertexts
//cout << "adding" << endl;
	seal::Ciphertext r = evaluate.add_many(t);
//	cout << "adding done" << endl;

	return r;
}

// loop over all rows: prediction for one row
seal::Ciphertext * calculate_predictions(int n_row, int n_col,
		seal::Ciphertext **x, seal::Ciphertext **theta,
		seal::Evaluator evaluate) {
//	cout << "caculat oredict"
//			<< x[0][0].operator seal::BigPolyArray &().coeff_count() << " "
//			<< x[0][0].operator seal::BigPolyArray &().coeff_bit_count()
//			<< endl;
	seal::Ciphertext* predictions = new seal::Ciphertext[n_col];
//	cout << "inside calculate pred" << endl;
// calculate h for each training example
	seal::Ciphertext tmp = h(x[0], theta, n_row, evaluate);
//	cout << "first one worked" << endl;
	*predictions = tmp;
//	cout << "allocation worked" << endl;
	for (int i = 1; i < n_col; i++) {
		//	cout << i << endl;
		//	Ciphertext* tmp = get_column(x, i, n_row, n_col);
		//	cout<< "getcolumn works for "
		*(predictions + i) = h(x[i], theta, n_row, evaluate);
		//	cout << "compute h" << endl;
//		cout << predictions[i].operator seal::BigPolyArray &().coeff_count()
//				<< " in calc "
//				<< predictions[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;
	}
	return predictions;
}
//paramter: pre initialized theta with 1
//perform gradien descent
seal::Ciphertext ** gradient_descent(int n_col, seal::Ciphertext **theta,
		seal::Ciphertext **x, seal::Ciphertext y[], Plaintext alpha, int iters,
		seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
		Plaintext text) {
	cout << "in gradient descent" << endl;

	for (int i = 0; i < iters; i++) {
		//	cout << i << endl;
		seal::Ciphertext *predictions = calculate_predictions(n_row, n_col, x,
				theta, evaluate);
//		cout << "made prediction " << i << endl;
		// h(x-y)
		seal::Ciphertext diff[n_col];
		for (int j = 0; j < n_col; j++) {
			diff[j] = evaluate.sub(predictions[j], y[j]);
			//	cout << "calculated difference " << j << endl;
		}
		seal::Ciphertext res[n_row + 1];
		seal::Ciphertext r[n_row + 1];
		// Plaintext plain = 1/n_row
		for (int k = 0; k <= n_row; k++) {
			for (int j = 0; j < n_col; j++) {
				res[k] =
						seal::Ciphertext(
								evaluate.relinearize(
										seal::Ciphertext(
												evaluate.multiply(
														diff[k].operator const seal::BigPolyArray &(),
														x[j][k].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
				/*seal::Ciphertext(
				 evaluate.multiply(
				 diff[k].operator const seal::BigPolyArray &(),
				 x[j][k].operator const seal::BigPolyArray &())); */
				/*		cout << diff[k].operator seal::BigPolyArray &().size()
				 << " and "
				 */
				//	cout << x[j][k].size() << endl;
				//	cout << res[k].size() << endl;
				/* cout << res[k].operator seal::BigPolyArray &().size();
				 *//*seal::Ciphertext(
				 evaluate.relinearize(
				 seal::Ciphertext(
				 evaluate.multiply(
				 diff[k].operator const seal::BigPolyArray &(),
				 x[j][k].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
				 */
				//	cout << "calculated multiply" << endl;
				r[k] =
						seal::Ciphertext(
								evaluate.relinearize(
										seal::Ciphertext(
												evaluate.multiply_plain(
														res[k].operator const seal::BigPolyArray &(),
														alpha)).operator const seal::BigPolyArray &()));
				//		cout << "calculated multiply plain" << endl;
				//cout << r[k].size() << endl;
				theta[k][0] =
						seal::Ciphertext(
								evaluate.sub(
										theta[k][0].operator const seal::BigPolyArray &(),
										r[k].operator const seal::BigPolyArray &()));
				//	cout << "calculated rest " << k << " + " << j << endl;
			}
		}
//		cout << "before compute cost" << endl;
		J[i] = compute_cost(n_row, n_col, x, y, theta, evaluate, text);
		//	cout << "calculated J " << endl;
	}
	return theta;
}

int main() {

//set parameters
	seal::EncryptionParameters parms;
	parms.set_poly_modulus("1x^4096 + 1");
	parms.set_coeff_modulus(
			seal::ChooserEvaluator::default_parameter_options().at(4096));
	parms.set_plain_modulus(1 << 8);
	parms.set_decomposition_bit_count(58);
	parms.validate();
//	cout << parms.poly_modulus().coeff_count() << endl;
//	cout << parms.poly_modulus().coeff_bit_count() << endl;
//read in data and change it to fitting format
	arma::mat data;
	data::Load("o-ring-erosion-or-blowby.csv", data);
	arma::vec responses;

// extract the last row of the data by transposing the matrix & extracting the second column
	arma::mat temp = data.t();
	responses = temp.col(1);
	data.shed_row(1);

//number of columns and rows
	int col = static_cast<int>(data.n_cols);
	int row = static_cast<int>(data.n_rows);

//Generate Keys
	seal::KeyGenerator generator(parms);
	generator.generate();
	const seal::BigPolyArray p_key = generator.public_key();
	seal::Ciphertext public_key = seal::Ciphertext(p_key);
	const seal::BigPoly s_key = generator.secret_key();
	seal::Plaintext secret_key = seal::Plaintext(s_key);
	generator.generate_evaluation_keys(col);
	const seal::EvaluationKeys evkey = generator.evaluation_keys();

//stores encrypted data
	seal::Ciphertext **encoded = new seal::Ciphertext*[row + 1];
//stores encrypted targets
	seal::Ciphertext resp[col];
//stores the weights
	seal::Ciphertext thet[col];
//stores decryption of prediction
	double *dec = new double[col];

//	transpose data matrix
	double **trans = new double*[row + 1];
	for (int i = 0; i < col; i++) {
		trans[i] = new double[row + 1];
		trans[i][0] = 1;
		for (int j = 1; j <= row; j++) {
			trans[i][j] = data(j - 1, i);
		}
	}
//	cout << "trans worked" << endl;
	seal::FractionalEncoder frencoder(parms.plain_modulus(),
			parms.poly_modulus(), 64, 32, 3);
	seal::IntegerEncoder encoder(parms.plain_modulus());
	const seal::EncryptionParameters par = parms;
	seal::Evaluator evaluate(par, evkey);
//seal::Evaluator evaluate(par);
	seal::Encryptor encryptor(parms, p_key);
	seal::Decryptor decryptor(par, s_key);

// encrypt targets y (1 x columns)
	for (int i = 0; i < col; i++) {
		encoded[i] = new seal::Ciphertext[row + 1];
		resp[i] = encrypt_frac(responses[i], encryptor, frencoder);
	}

//	cout << "resp enc worked" << endl;

// split data into training and data set and encrypt it
	double **train_unencr = new double*[row + 1];
	double **data_unencr = new double*[row + 1];
//stores encrypted train set
	seal::Ciphertext **train_set = new seal::Ciphertext*[row + 1];
//stores encrypted data set
	seal::Ciphertext **data_set = new seal::Ciphertext*[row + 1];

//encryption and split in one step
//error: j=5, i=0 bus error
	/*	for (int j = 0; j <= col / 2; j++) {
	 data_set[j] = new seal::Ciphertext[row + 1];
	 for (int i = 0; i <= row; i++) {
	 //	cout << j << " and " << i << endl;
	 seal: Ciphertext tmp = encrypt_frac(trans[j + (col / 2)][i],
	 encryptor, frencoder);
	 //	cout << "init alright" << endl;
	 data_set[j][i] = tmp;
	 }
	 }

	 //error: j=5, i=0: free(): invalid size: 0x000055c97d8a0ef0
	 for (int j = 0; j < col / 2; j++) {
	 train_set[j] = new seal::Ciphertext[row + 1];
	 for (int i = 0; i <= row; i++) {
	 //	cout << j << " andd " << i << endl;
	 Ciphertext tmp = encrypt_frac(trans[j][i], encryptor, frencoder);
	 //	cout << "init fine" << endl;
	 train_set[j][i] = tmp;
	 }
	 } */

// encryption and split in single steps
//error:
	/*	for (int j = 0; j < col / 2; j++) {
	 train_unencr[j] = new double[row + 1];
	 for (int i = 0; i <= row; i++) {
	 train_unencr[j][i] = trans[j][i];
	 }
	 }
	 for (int j = 0; j <= col / 2; j++) {
	 data_unencr[j] = new double[row + 1];
	 for (int i = 0; i <= row; i++) {
	 data_unencr[j][i] = trans[j + (col / 2)][i];
	 }
	 }
	 //error: j=5, i=0 free(): invalid size 0x00005570ed1a5ef0
	 for (int j = 0; j < col / 2; j++) {
	 train_set[j] = new seal::Ciphertext[row + 1];
	 for (int i = 0; i <= row; i++) {
	 cout << j << " and " << i << endl;
	 seal::Ciphertext t = encrypt_frac(train_unencr[j][i], encryptor,
	 frencoder);
	 cout << "encryption fine" << endl;
	 train_set[j][i] = t;
	 cout << "train encrypt" << endl;
	 }
	 }
	 //error: j=9, i=0 segmentation fault
	 for (int j = 0; j <= col / 2; j++) {
	 data_set[j] = new seal::Ciphertext[row + 1];
	 for (int i = 0; i <= row; i++) {
	 cout << j << " andd " << i << endl;
	 seal::Ciphertext tmp = encrypt_frac(data_unencr[j][i], encryptor,
	 frencoder);
	 cout << "init alright" << endl;
	 data_set[j][i] = tmp;
	 }
	 } */

//	cout << "encrypted worked" << endl;
// encrypt complete data set (column x row)
	for (int i = 0; i < col; i++) {
		for (int j = 0; j <= row; j++) {
			encoded[i][j] = encrypt_frac(trans[i][j], encryptor, frencoder);
		}
	}
//	cout << "encoding worked" << endl;
//learning rate alpha
	seal::Plaintext alpha = frencoder.encode(0.32);
//constant (1.0 / (2 * n_col)
	seal::Plaintext text = frencoder.encode((1.0 / (2 * col)));

//#columns, dataset, theta, Evaluator
//	cout << "before theta" << endl;
// initialize theta with 1 and encrypt it (row x 1)
	seal::Ciphertext tmp = encrypt_frac(1.0, encryptor, frencoder);

	seal::Ciphertext **theta = new seal::Ciphertext*[1];
	for (int i = 0; i <= row; i++) {
//		cout << i << endl;
		theta[i] = new seal::Ciphertext[1];
		theta[i][0] = tmp;
	}
//	cout << theta[1][0].operator seal::BigPolyArray &().coeff_count() << " ! "
//			<< theta[1][0].operator seal::BigPolyArray &().coeff_bit_count()
//			<< endl;
//	cout << "theta worked" << endl;
	cout << "before train" << endl;
	/* train
	 parameters: learning rate alpha, #iterations, number of columns,
	 encrypted data, encoded result, number of rows, encoded constant, evaluator */
	Ciphertext** trained = train(alpha, 1, col, encoded, resp, theta, row, text,
			evaluate);
	cout << "after train" << endl;
	seal::Ciphertext** weights = new seal::Ciphertext*[1];
	weights = trained;
	cout << "pred" << endl;
	/*make predicitons on data
	 parameters: number of rows, number of columns, encrypted data,
	 encrypted weights, evaluator */

	seal::Ciphertext* pred = predict(row, col, encoded, weights, evaluate);
	cout << "prediction in main" << endl;

	for (int i = 0; i < col; i++) {
		cout << i << endl;
		double d = decrypt_frac((*(pred + i)), decryptor, frencoder);
		//cout << "is decrypt" << endl;
		dec[i] = d;
		//cout << "decr works" << endl;
		cout << dec[i] << endl;
	}
	cout << "decryption worked" << endl;
// prediction on unencrypted data
	arma::vec preds = reg_lin_reg(data, responses, 0.0);
	for (int i = 0; i < col; i++) {
		cout << preds[i] << endl;
	}
}
