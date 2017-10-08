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
#include <cstdlib>
#include <cstdio>
#include </usr/include/mlpack/core.hpp>
//#include </usr/include/mlpack/methods/linear_regression/linear_regression.hpp>
#include "Seal/seal.h"
#include "Seal/plaintext.h"
#include "Seal/ciphertext.h"
#include "Seal/encryptor.h"
#include "Seal/encryptionparams.h"
#include "Seal/keygenerator.h"
#include "Seal/evaluationkeys.h"
#include "Seal/bigpoly.h"
#include "Seal/biguint.h"
#include "Seal/bigpolyarray.h"
#include "Seal/chooser.h"
#include "Seal/evaluator.h"
#include <chrono>
//#include "LogisticRegression.h"
// error with constructor when using header file
#include "LogisticRegression.cpp"
//#include "OwnLinearRegression.h"
#include "OwnLinearRegression.cpp"

//TODO inclusion of header files; deletion of unsued arrays, error: corrputed size vs. previous size in training methods as well as prediction in multiclass classification


using namespace std;
using namespace seal;
using namespace mlpack;
//using namespace mlpack::regression;

//encrypts an integer by turning it into Plaintext (polynomial) and then Ciphertext
Ciphertext encrypt(const int input, Encryptor encryptor,
		IntegerEncoder encoder) {
	Plaintext encoded = encoder.encode(input);
	Ciphertext encrypted = encryptor.encrypt(encoded);
	return encrypted;
}

// encrypt a double
Ciphertext encrypt_frac(double input, Encryptor encryptor,
		FractionalEncoder encoder) {
	Plaintext encoded = encoder.encode(input);
	Ciphertext encrypted = encryptor.encrypt(encoded);
	return encrypted;
}

// decrypts given ciphertext by turning it into Plaintext and then integer
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
	//cout << "decryptor" << endl;
	BigPoly decrypted = decryptor.decrypt(
			encrypted.operator const seal::BigPolyArray &());
//	cout << "Plaintext" << endl;
	double decryptt = encoder.decode(decrypted);
//	cout << "int" << endl;
	return decryptt;
	// Decode results.
}

////perform linear regression

//arma::vec reg_lin_reg(arma::mat data, arma::vec responses, double lambda) {
//
////	// perform linear regression; lambda = 0.5 for ridge regression; 0.0 for linear regression
//	LinearRegression lr(data, responses, lambda);
//	arma::vec parameters = lr.Parameters();
////
//	arma::mat train_data = data.submat(0, 0, 3, 11);
//	arma::vec train_target = responses.subvec(0, 11);
//	arma::mat test_data = data.submat(0, 11, 3, 22);
//	arma::vec test_target = responses.subvec(11, 22);
////	lr.Train(train_data, train_target, false, );
//	arma::vec predictions;
//	lr.Predict(data, predictions);
//	return predictions;
//}

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

//uncomment when running binary logistic regression///////
/*  int main(){

 //set parameters for encryption
 seal::EncryptionParameters parms;
 parms.set_poly_modulus("1x^4096 + 1");
 parms.set_coeff_modulus(
 seal::ChooserEvaluator::default_parameter_options().at(4096));
 parms.set_plain_modulus(1 << 8);
 parms.set_decomposition_bit_count(58);
 parms.validate();

 // logistic regression
 //read in data for classification
 arma::mat train_dat_reg;
 arma::vec resp_train_reg;
 arma::mat test_dat_reg;
 arma::vec resp_test_reg;
 data::Load("spect_train.csv", train_dat_reg);
 data::Load("spect_test.csv", test_dat_reg);
 arma::mat tmp_reg = train_dat_reg.t();
 arma::mat tmp2_reg = test_dat_reg.t();
 resp_train_reg = tmp_reg.col(0);
 resp_test_reg = tmp2_reg.col(0);
 train_dat_reg.shed_row(0);
 test_dat_reg.shed_row(0);

 //number of columns and rows for training and test set
 int train_col_reg = static_cast<int>(train_dat_reg.n_cols);
 int train_row_reg = static_cast<int>(train_dat_reg.n_rows);
 int test_col_reg = static_cast<int>(test_dat_reg.n_cols);
 int test_row_reg = static_cast<int>(test_dat_reg.n_rows);

 //	transpose data matrices
 double **train_dat_trans_reg = new double*[train_row_reg + 1];
 double ** test_dat_trans_reg = new double*[test_row_reg + 1];
 for (int i = 0; i < train_col_reg; i++) {
 train_dat_trans_reg[i] = new double[train_row_reg + 1];
 train_dat_trans_reg[i][0] = 1;
 for (int j = 1; j <= train_row_reg; j++) {
 train_dat_trans_reg[i][j] = train_dat_reg(j - 1, i);
 }
 }
 for (int i = 0; i < test_col_reg; i++) {
 test_dat_trans_reg[i] = new double[test_row_reg + 1];
 test_dat_trans_reg[i][0] = 1;
 for (int j = 1; j <= test_row_reg; j++) {
 test_dat_trans_reg[i][j] = test_dat_reg(j - 1, i);
 }
 }

 //Generate Keys
 seal::KeyGenerator generator(parms);
 generator.generate();
 const seal::BigPolyArray p_key = generator.public_key();
 seal::Ciphertext public_key = seal::Ciphertext(p_key);
 const seal::BigPoly s_key = generator.secret_key();
 seal::Plaintext secret_key = seal::Plaintext(s_key);
 generator.generate_evaluation_keys(test_col_reg);
 const seal::EvaluationKeys evkey = generator.evaluation_keys();
 seal::FractionalEncoder frencoder(parms.plain_modulus(),
 parms.poly_modulus(), 64, 32, 3);
 seal::IntegerEncoder encoder(parms.plain_modulus());
 seal::Evaluator evaluate(parms, evkey);
 seal::Encryptor encryptor(parms, p_key);
 seal::Decryptor decryptor(parms, s_key);

 //#columns, dataset, theta, Evaluator

 seal::Ciphertext ** encoded_train_reg = new seal::Ciphertext*[train_row_reg
 + 1];
 seal::Ciphertext ** encoded_test_reg = new seal::Ciphertext*[test_row_reg
 + 1];
 seal::Ciphertext* test_resp_reg = new seal::Ciphertext[test_col_reg];
 seal::Ciphertext* train_resp_reg = new seal::Ciphertext[train_col_reg];

 auto start = chrono::steady_clock::now();
 //	encrypt targets y (1 x columns)
 for (int i = 0; i < train_col_reg; i++) {
 Ciphertext t = encrypt_frac(resp_train_reg(i), encryptor, frencoder);
 train_resp_reg[i] = t;
 }
 auto end = chrono::steady_clock::now();
 auto diffr = end - start;
 cout << "encrypt train targets "
 << chrono::duration<double, ratio<1>>(diffr).count() << " s"
 << endl;

 start = chrono::steady_clock::now();
 for (int i = 0; i < test_col_reg; i++) {
 test_resp_reg[i] = encrypt_frac(resp_test_reg(i), encryptor, frencoder);
 }
 end = chrono::steady_clock::now();
 diffr = end - start;
 cout << "encrypt test targets "
 << chrono::duration<double, ratio<1>>(diffr).count() << " s"
 << endl;

 start = chrono::steady_clock::now();
 for (int i = 0; i < train_col_reg; i++) {
 encoded_train_reg[i] = new seal::Ciphertext[train_row_reg + 1];
 for (int j = 0; j <= train_row_reg; j++) {
 encoded_train_reg[i][j] = encrypt_frac(train_dat_trans_reg[i][j],
 encryptor, frencoder);
 }
 }
 end = chrono::steady_clock::now();
 diffr = end - start;
 cout << "encrypt train data "
 << chrono::duration<double, ratio<1>>(diffr).count() << " s"
 << endl;

 // deleting unused transpose of train matrix
 // deleting of Ciphertext arrays is not working anywhere at the moment
// for (int i = 0; i <= train_col_reg; i++) {
// delete[] train_dat_trans_reg[i];
// }
 //	delete[] train_dat_trans_reg;
 start = chrono::steady_clock::now();

 for (int i = 0; i < test_col_reg; i++) {
 encoded_test_reg[i] = new seal::Ciphertext[test_row_reg + 1];
 for (int j = 0; j <= test_row_reg; j++) {
 encoded_test_reg[i][j] = encrypt_frac(test_dat_trans_reg[i][j],
 encryptor, frencoder);
 }
 }
 end = chrono::steady_clock::now();
 diffr = end - start;
 cout << "encrypt test data "
 << chrono::duration<double, ratio<1>>(diffr).count() << " s"
 << endl;

 //deleting unused transpose of test matrix
 //	for (int i = 0; i <= test_row_reg; i++) {
 //		delete[] test_dat_trans_reg[i];
 //	}
 //	delete[] test_dat_trans_reg;


 // training rate alpha
 double alph_reg = 0.32;
 // tr = lambda/2*col, lambda security parameter?
 double lambda_reg = 0.4;
 seal::Plaintext fr = frencoder.encode(1 / (1.0 * train_col_reg));
 seal::Plaintext a0 = frencoder.encode(-0.714761);
 seal::Plaintext a1 = frencoder.encode(-0.5);
 seal::Plaintext a2 = frencoder.encode(-0.0976419);
 seal::Plaintext minus_one_reg = frencoder.encode(-1.0);
 seal::Plaintext tr = frencoder.encode(lambda_reg / (2.0 * train_col_reg));
 seal::Plaintext alpha_reg = frencoder.encode(alph_reg);
 LogisticRegression lr; // = LogisticRegression::LogisticRegression();



 // initialize theta for regression and classification with 1 and encrypt it (row x 1)
 seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
 seal::Ciphertext **theta_reg = new seal::Ciphertext*[1];

 for (int i = 0; i <= test_row_reg; i++) {
 theta_reg[i] = new seal::Ciphertext[1];

 theta_reg[i][0] = one_encrypt;

 }


 // train the classification model
 start = chrono::steady_clock::now();
 seal::Ciphertext** weights_reg = lr.train(1, train_col_reg, train_row_reg,
 encoded_train_reg, theta_reg, train_resp_reg, evaluate, fr, a0, a1,
 a2, minus_one_reg, tr, alpha_reg);
 end = chrono::steady_clock::now();
 diffr = end - start;
 cout << "train" << chrono::duration<double, ratio<60>>(diffr).count()
 << " min" << endl;

 //delete pre-initialized theta
 //	delete[] theta_reg[0];
 //	delete[] theta_reg;

 start = chrono::steady_clock::now();

// //make binary class predictions
 seal::Ciphertext* predicitons_reg = lr.predict(test_col_reg, test_row_reg,
 encoded_test_reg, weights_reg, a0, a1, a2, evaluate);
 end = chrono::steady_clock::now();
 diffr = end - start;
 cout << "predicition" << chrono::duration<double, ratio<60>>(diffr).count()
 << " min" << endl;

 //cout << "predict fone" << endl;
 double * predictions_reg_encrypted = new double[test_col_reg];
 int * class_prediction = new int[test_col_reg];
 start = chrono::steady_clock::now();
 for (int i = 0; i < test_col_reg; i++) {
 predictions_reg_encrypted[i] = decrypt_frac(predicitons_reg[i],
 decryptor, frencoder);
 if (predictions_reg_encrypted[i] < 0.5) {
 class_prediction[i] = 0;
 } else {
 class_prediction[i] = 1;
 }
 //cout << i << endl;
 cout << class_prediction[i] << endl;
 }
 end = chrono::steady_clock::now();
 diffr = end - start;
 cout << "decrypt test data "
 << chrono::duration<double, ratio<1>>(diffr).count() << " s"
 << endl;
 return 0;
 } */

// uncomment when running multiclass logistic regression
 int main() {
	//uncomment when run logistic regression///////
//set parameters for encryption
	seal::EncryptionParameters parms;
	parms.set_poly_modulus("1x^4096 + 1");
	parms.set_coeff_modulus(
			seal::ChooserEvaluator::default_parameter_options().at(4096));
	parms.set_plain_modulus(1 << 8);
	parms.set_decomposition_bit_count(58);
	parms.validate();

// logistic regression
//read in data for classification
	//arma::mat train_dat_reg;
//	arma::vec resp_train_reg;
	arma::mat test_dat_reg;
	arma::vec resp_test_reg;
//	data::Load("spect_train.csv", train_dat_reg);
	data::Load("iris.csv", test_dat_reg);
//	arma::mat tmp_reg = train_dat_reg.t();
	arma::mat tmp2_reg = test_dat_reg.t();
//	resp_train_reg = tmp_reg.col(0);
	resp_test_reg = tmp2_reg.col(0);
//	train_dat_reg.shed_row(0);
	test_dat_reg.shed_row(0);

//number of columns and rows for training and test set
//	int train_col_reg = static_cast<int>(train_dat_reg.n_cols);
//	int train_row_reg = static_cast<int>(train_dat_reg.n_rows);
	int test_col_reg = static_cast<int>(test_dat_reg.n_cols);
	int test_row_reg = static_cast<int>(test_dat_reg.n_rows);

//	transpose data matrices
//	double **train_dat_trans_reg = new double*[train_row_reg + 1];
	double ** test_dat_trans_reg = new double*[test_row_reg + 1];
//	for (int i = 0; i < train_col_reg; i++) {
//		train_dat_trans_reg[i] = new double[train_row_reg + 1];
//		train_dat_trans_reg[i][0] = 1;
//		for (int j = 1; j <= train_row_reg; j++) {
//			train_dat_trans_reg[i][j] = train_dat_reg(j - 1, i);
//		}
//	}
	for (int i = 0; i < test_col_reg; i++) {
		test_dat_trans_reg[i] = new double[test_row_reg + 1];
		test_dat_trans_reg[i][0] = 1;
		for (int j = 1; j <= test_row_reg; j++) {
			test_dat_trans_reg[i][j] = test_dat_reg(j - 1, i);
		}
	}

//Generate Keys
	seal::KeyGenerator generator(parms);
	generator.generate();
	const seal::BigPolyArray p_key = generator.public_key();
	seal::Ciphertext public_key = seal::Ciphertext(p_key);
	const seal::BigPoly s_key = generator.secret_key();
	seal::Plaintext secret_key = seal::Plaintext(s_key);
	generator.generate_evaluation_keys(test_col_reg);
	const seal::EvaluationKeys evkey = generator.evaluation_keys();
	seal::FractionalEncoder frencoder(parms.plain_modulus(),
			parms.poly_modulus(), 64, 32, 3);
	seal::IntegerEncoder encoder(parms.plain_modulus());
	seal::Evaluator evaluate(parms, evkey);
//seal::Evaluator evaluate(par);
	seal::Encryptor encryptor(parms, p_key);
	seal::Decryptor decryptor(parms, s_key);

//#columns, dataset, theta, Evaluator

//	seal::Ciphertext ** encoded_train_reg = new seal::Ciphertext*[train_row_reg
//			+ 1];
	seal::Ciphertext ** encoded_test_reg = new seal::Ciphertext*[test_row_reg
			+ 1];
	seal::Ciphertext* test_resp_reg = new seal::Ciphertext[test_col_reg];
	//seal::Ciphertext* train_resp_reg = new seal::Ciphertext[train_col_reg];

//	auto start = chrono::steady_clock::now();
////	encrypt targets y (1 x columns)
//	for (int i = 0; i < train_col_reg; i++) {
//		Ciphertext t = encrypt_frac(resp_train_reg(i), encryptor, frencoder);
//		train_resp_reg[i] = t;
//	}
//	auto end = chrono::steady_clock::now();
//	auto diffr = end - start;
//	cout << "encrypt train targets "
//			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
//			<< endl;
//cout << "train response" << endl;
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < test_col_reg; i++) {
		test_resp_reg[i] = encrypt_frac(resp_test_reg(i), encryptor, frencoder);
	}
	auto end = chrono::steady_clock::now();
	auto diffr = end - start;
	cout << "encrypt test targets "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

//	start = chrono::steady_clock::now();
////cout << "test response" << endl;
//	for (int i = 0; i < train_col_reg; i++) {
//		encoded_train_reg[i] = new seal::Ciphertext[train_row_reg + 1];
//		for (int j = 0; j <= train_row_reg; j++) {
//			encoded_train_reg[i][j] = encrypt_frac(train_dat_trans_reg[i][j],
//					encryptor, frencoder);
//		}
//	}
//	end = chrono::steady_clock::now();
//	diffr = end - start;
//	cout << "encrypt train data "
//			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
//			<< endl;


//deleting unused transpose of train matrix
//	for (int i = 0; i <= train_col_reg; i++) {
//		delete[] train_dat_trans_reg[i];
//	}
//	delete[] train_dat_trans_reg;
	start = chrono::steady_clock::now();
	for (int i = 0; i < test_col_reg; i++) {
		encoded_test_reg[i] = new seal::Ciphertext[test_row_reg + 1];
		for (int j = 0; j <= test_row_reg; j++) {
			encoded_test_reg[i][j] = encrypt_frac(test_dat_trans_reg[i][j],
					encryptor, frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

//deleting unused transpose of test matrix
//	for (int i = 0; i <= test_row_reg; i++) {
//		delete[] test_dat_trans_reg[i];
//	}
//	delete[] test_dat_trans_reg;


// training rate alpha
	double alph_reg = 0.32;
// tr = lambda/2*col, lambda security parameter?
	double lambda_reg = 0.4;
//	seal::Plaintext fr = frencoder.encode(1 / (1.0 * train_col_reg));
	seal::Plaintext a0 = frencoder.encode(-0.714761);
	seal::Plaintext a1 = frencoder.encode(-0.5);
	seal::Plaintext a2 = frencoder.encode(-0.0976419);
	seal::Plaintext minus_one_reg = frencoder.encode(-1.0);
//	seal::Plaintext tr = frencoder.encode(lambda_reg / (2.0 * train_col_reg));
	seal::Plaintext alpha_reg = frencoder.encode(alph_reg);
	LogisticRegression lr; // = LogisticRegression::LogisticRegression();


// initialize theta for regression and classification with 1 and encrypt it (row x 1)
	seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
	seal::Ciphertext **theta_reg = new seal::Ciphertext*[1];

	for (int i = 0; i <= test_row_reg; i++) {
		theta_reg[i] = new seal::Ciphertext[1];

		theta_reg[i][0] = one_encrypt;

	}

// train the classification model
//	start = chrono::steady_clock::now();
//	seal::Ciphertext** weights_reg = lr.train(1, train_col_reg, train_row_reg,
//			encoded_train_reg, theta_reg, train_resp_reg, evaluate, fr, a0, a1,
//			a2, minus_one_reg, tr, alpha_reg);
//	end = chrono::steady_clock::now();
//	diffr = end - start;
//	cout << "train" << chrono::duration<double, ratio<60>>(diffr).count()
//			<< " min" << endl;

//delete initialized theta
//	delete[] theta_reg[0];
//	delete[] theta_reg;

	start = chrono::steady_clock::now();
//make multiclass predictions

	int k = 3;
//	seal::Ciphertext** predicitons_reg = lr.predict_multiclass(k, test_col_reg,
//			test_row_reg, encoded_test_reg, weights_reg, a0, a1, a2, evaluate);
	seal::Ciphertext** predicitons_reg = lr.predict_multiclass(k, test_col_reg,
			test_row_reg, encoded_test_reg, theta_reg, a0, a1, a2, evaluate);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "predicition" << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;

	double ** predictions_reg_encrypted = new double*[test_col_reg];
	int * class_prediction = new int[test_col_reg];
	start = chrono::steady_clock::now();
	for (int j = 0; j < k; j++) {
		predictions_reg_encrypted[j] = new double[test_col_reg];
		for (int i = 0; i < test_col_reg; i++) {
			predictions_reg_encrypted[j][i] = decrypt_frac(
					predicitons_reg[j][i], decryptor, frencoder);

		}
	}

	// make multiclass predictions: take maximum of all the classifiers (One-vs-all)
	for (int i = 0; i < test_col_reg; i++) {
		double max = predictions_reg_encrypted[i][0];
		int cl = 0;
		for (int j = 1; j < k; j++) {
			if (predictions_reg_encrypted[i][j] > max) {
				max = predictions_reg_encrypted[i][j];
				cl = j;
			}
			if (j == k - 1) {
				class_prediction[i] = cl;
			}
		}
		cout << class_prediction[i] << endl;
	}

	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "decrypt test data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	start = chrono::steady_clock::now();
return 0;
}

//uncomment when running linear regression
/* int main() {
	// set parameters for encryption
	seal::EncryptionParameters parms;
	parms.set_poly_modulus("1x^4096 + 1");
	parms.set_coeff_modulus(
			seal::ChooserEvaluator::default_parameter_options().at(4096));
	parms.set_plain_modulus(1 << 8);
	parms.set_decomposition_bit_count(58);
	parms.validate();

	//Generate Keys for encryption
	seal::KeyGenerator generator(parms);
	generator.generate();
	const seal::BigPolyArray p_key = generator.public_key();
	seal::Ciphertext public_key = seal::Ciphertext(p_key);
	const seal::BigPoly s_key = generator.secret_key();
	seal::Plaintext secret_key = seal::Plaintext(s_key);

//read in data for classification
	arma::mat train_dat;
	arma::vec resp_train;
	arma::mat test_dat;
	arma::vec resp_test;
	data::Load("parkinson_train.csv", train_dat);
	data::Load("parkinson_test.csv", test_dat);
	arma::mat tmp = train_dat.t();
	arma::mat tmp2 = test_dat.t();
	resp_train = tmp.col(5); // label of train data
	resp_test = tmp2.col(5);

//	train_dat.shed_row(train_dat.n_rows - 1);
//	test_dat.shed_row(test_dat.n_rows - 1);
//
//	train_dat.shed_row(train_dat.n_rows - 3);
//	test_dat.shed_row(test_dat.n_rows - 3);
//	train_dat.shed_row(17);
//	train_dat.shed_row(17);
//	train_dat.shed_row(16);
//	test_dat.shed_row(16);
//	train_dat.shed_row(15);
//	test_dat.shed_row(15);
//	train_dat.shed_row(14);
//	test_dat.shed_row(14);
//	train_dat.shed_row(13);
//	test_dat.shed_row(13);
//
//	test_dat.shed_row(12);
//	test_dat.shed_row(12);
//	train_dat.shed_row(10);
//	test_dat.shed_row(10);
//	train_dat.shed_row(9);
//	test_dat.shed_row(9);
//
//	train_dat.shed_row(7);
//	test_dat.shed_row(7);
//
//	train_dat.shed_row(6);
//	test_dat.shed_row(6);
//
//	train_dat.shed_row(5);
//	test_dat.shed_row(5);
	int train_row = static_cast<int>(train_dat.n_rows);
	for (int i = train_row - 1; i > 4; i--) {
		train_dat.shed_row(i);
		test_dat.shed_row(i);

	}
	train_dat.shed_row(0);
	test_dat.shed_row(0);

//number of columns and rows for training and test set
	int train_col = static_cast<int>(train_dat.n_cols);
	train_row = static_cast<int>(train_dat.n_rows);
	int test_col = static_cast<int>(test_dat.n_cols);
	int test_row = static_cast<int>(test_dat.n_rows);

	// generate keys in order to relinearize ciphertexts
	generator.generate_evaluation_keys(train_col);
	const seal::EvaluationKeys evkey = generator.evaluation_keys();
	seal::FractionalEncoder frencoder(parms.plain_modulus(),
			parms.poly_modulus(), 64, 32, 3);
	seal::IntegerEncoder encoder(parms.plain_modulus());
	seal::Evaluator evaluate(parms, evkey);
	//seal::Evaluator evaluate(par);
	seal::Encryptor encryptor(parms, p_key);
	seal::Decryptor decryptor(parms, s_key);

	// encoding and encrpypting the polynomials
	seal::Ciphertext **encoded_train = new seal::Ciphertext *[train_row + 1];
//	transpose data matrices
	double **train_dat_trans = new double *[train_row + 1];
	double **test_dat_trans = new double *[test_row + 1];
	for (int i = 0; i < train_col; i++) {
		train_dat_trans[i] = new double[train_row + 1];
		train_dat_trans[i][0] = 1;
		for (int j = 1; j <= train_row; j++) {
			train_dat_trans[i][j] = train_dat(j - 1, i);
		}
	}
//	for (int i = 0; i < train_col; i++) {
//		train_dat_trans[i] = NULL;
//		delete[] train_dat_trans[i];
//	}
//	delete[] train_dat_trans;
	for (int i = 0; i < test_col; i++) {
		test_dat_trans[i] = new double[test_row + 1];
		test_dat_trans[i][0] = 1;
		for (int j = 1; j <= test_row; j++) {
			test_dat_trans[i][j] = test_dat(j - 1, i);
		}
	}

//const seal::EncryptionParameters par = parms;


	// encrypt the given data

	// encrypt train data
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < train_col; i++) {
		encoded_train[i] = new seal::Ciphertext[train_row + 1];
		for (int j = 0; j <= train_row; j++) {
			seal::Ciphertext t = encrypt_frac(train_dat_trans[i][j], encryptor,
					frencoder);
			encoded_train[i][j] = t;
		}
	}
	auto end = chrono::steady_clock::now();
	auto diffr = end - start;
	cout << "encrypt train data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;
	seal::Ciphertext *train_resp = new seal::Ciphertext[train_col];
	start = chrono::steady_clock::now();

//	encrypt train labels y (1 x columns)
	for (int i = 0; i < train_col; i++) {
		Ciphertext t = encrypt_frac(resp_train(i), encryptor, frencoder);
		train_resp[i] = t;

	}

	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt train targets "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;
//	seal::Ciphertext *test_train_resp = new seal::Ciphertext[train_col];
//	// temporary soluztion as something changes values inside array
//	for (int i = 0; i < train_col; i++) {
//		test_train_resp[i] = train_resp[i];
//	}
// encrypt test labels
	seal::Ciphertext *test_resp = new seal::Ciphertext[test_col];
	start = chrono::steady_clock::now();
	for (int i = 0; i < test_col; i++) {
		seal::Ciphertext t = encrypt_frac(resp_test(i), encryptor, frencoder);
		test_resp[i] = t;
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test targets "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

//	for (int i = 0; i < train_col; i++) {
//		cout << i << endl;
//		delete[] train_dat_trans[i];
//	}
//	delete[] train_dat_trans;
//deleting unused transpose of train matrix
//	for (int i = 0; i <= train_col_reg; i++) {
//		cout << i << endl;
//		delete[] train_dat_trans_reg[i];
//	}
//	delete[] train_dat_trans_reg;

//	for (int i = 0; i < train_col; i++) {
//		cout << train_resp[i].operator seal::BigPolyArray &().coeff_count()
//				<< endl;
//		cout << train_resp[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;
//	}
	// encrypt test data
	seal::Ciphertext **encoded_test = new seal::Ciphertext *[test_row + 1];
	start = chrono::steady_clock::now();

	for (int i = 0; i < test_col; i++) {
		encoded_test[i] = new seal::Ciphertext[test_row + 1];
		for (int j = 0; j <= test_row; j++) {
			encoded_test[i][j] = encrypt_frac(test_dat_trans[i][j], encryptor,
					frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	//learning rate alpha = 0.32
	double alph = 0.032 / train_col;
	seal::Plaintext alpha = frencoder.encode(alph);
//	constant (1.0 / (2 * n_col)
	seal::Plaintext text = frencoder.encode((1.0 / (2.0 * train_col)));
	seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
	double lambda = 1.5 / (1.0 * train_col);
	seal::Plaintext lambda_div = frencoder.encode(lambda);
	// initialize the weights
	seal::Ciphertext **theta = new seal::Ciphertext *[1];
	for (int i = 0; i <= train_row; i++) {
		theta[i] = new seal::Ciphertext[1];
		theta[i][0] = one_encrypt;

	}

	bool ridge = true;

	OwnLinearRegression linreg;
//	start = chrono::steady_clock::now();
//	// train the linear regression model
//	Ciphertext **trained = linreg.train(alpha, 1, train_col, encoded_train,
//			train_resp, theta, train_row, text, evaluate, ridge, lambda_div);
//	end = chrono::steady_clock::now();
//	diffr = end - start;
//	cout << "train " << chrono::duration<double, ratio<60>>(diffr).count()
//			<< " min" << endl;
//	delete pre - initialized theta
//	delete[] theta[0];
//	delete[] theta;

//	seal::Ciphertext **weights = new seal::Ciphertext *[1];
//	weights = trained;

	//make predicitons on test data set
	start = chrono::steady_clock::now();
	seal::Ciphertext *pred = linreg.predict(test_row, test_col, encoded_test,
			weights, evaluate);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "pred " << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;
	cout << "prediction in main" << endl;
	double *dec = new double[test_col];
// later: compare prediction on unencrypted and encrypted data
	start = chrono::steady_clock::now();
	for (int i = 0; i < test_col; i++) {
		double d = decrypt_frac((*(pred + i)), decryptor, frencoder);
		dec[i] = d;
		cout << dec[i] << endl;
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	return 0;
}*/
