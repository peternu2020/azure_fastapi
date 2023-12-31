from fastapi.testclient import TestClient

from .main import app

test_client = TestClient(app)

def test_valid_post_response():
    response = test_client.post(
        "/predict",
        json={"x12":"6882.34",
              "x44":"0.98441208",
              "x53":"0.33612654",
              "x56":"0.137700027",
              "x58":"0.228881625",
              "x62": "0.373941237",
              "x91":"0.04191",
              "x5":"tuesday",
              "x31":"germany",
              "x81":"April"
              }
    )
    
    assert response.status_code == 200
    assert list(response.json()[0].keys()) == sorted(['business_outcome', 'phat', 'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', \
    'x81_October', 'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', \
    'x53', 'x81_November', 'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', \
    'x58', 'x56'])
    assert len(response.json()[0]) == 27 #27 fields expected in response; business_outcome, phat, and 25 of the model features
    assert response.json()[0]['business_outcome'] == 0
    assert round(response.json()[0]['phat'],2) == 0.09
    assert response.json()[0]['x31_germany'] == 1
    assert response.json()[0]['x5_tuesday'] == 1
    


def test_valid_post_response2():
  #variation of test_valid_post_response with the value in the x12 field being formatted as a number/float instead of a string
    response = test_client.post(
        "/predict",
        json={"x12":6882.34,
              "x44":"0.98441208",
              "x53":"0.33612654",
              "x56":"0.137700027",
              "x58":"0.228881625",
              "x62": "0.373941237",
              "x91":"0.04191",
              "x5":"tuesday",
              "x31":"germany",
              "x81":"April"
              }
    )
    assert response.status_code == 200
    assert list(response.json()[0].keys()) == sorted(['business_outcome', 'phat', 'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', \
    'x81_October', 'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', \
    'x53', 'x81_November', 'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', \
    'x58', 'x56'])
    assert len(response.json()[0]) == 27 #27 fields expected in response; business_outcome, phat, and 25 of the model features
    assert response.json()[0]['business_outcome'] == 0
    assert round(response.json()[0]['phat'],2) == 0.09
    assert response.json()[0]['x31_germany'] == 1
    assert response.json()[0]['x5_tuesday'] == 1
                                
def test_valid_post_response3():
  #test input with None value (valid-JSON)
    response = test_client.post(
        "/predict",
        json={"x12": None,
              "x44":"0.98441208",
              "x53":"0.33612654",
              "x56":"0.137700027",
              "x58":"0.228881625",
              "x62": "0.373941237",
              "x91":"0.04191",
              "x5":"tuesday",
              "x31":"germany",
              "x81":"April"
              }
    )
    assert response.status_code == 200
    assert list(response.json()[0].keys()) == sorted(['business_outcome', 'phat', 'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', \
    'x81_October', 'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', \
    'x53', 'x81_November', 'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', \
    'x58', 'x56'])
    assert len(response.json()[0]) == 27 #27 fields expected in response; business_outcome, phat, and 25 of the model features
    assert response.json()[0]['business_outcome'] in [0, 1] 
    assert response.json()[0]['x31_germany'] == 1
    assert response.json()[0]['x5_tuesday'] == 1
      
      
def test_valid_post_response4():
  #test input with empty string value (valid-JSON)
    response = test_client.post(
        "/predict",
        json={"x12": "",
              "x44":"0.98441208",
              "x53":"0.33612654",
              "x56":"0.137700027",
              "x58":"0.228881625",
              "x62": "0.373941237",
              "x91":"0.04191",
              "x5":"tuesday",
              "x31":"germany",
              "x81":"April"
              }
    )
    assert response.status_code == 200
    assert list(response.json()[0].keys()) == sorted(['business_outcome', 'phat', 'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', \
    'x81_October', 'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', \
    'x53', 'x81_November', 'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', \
    'x58', 'x56'])
    assert len(response.json()[0]) == 27 #27 fields expected in response; business_outcome, phat, and 25 of the model features
    assert response.json()[0]['business_outcome'] in [0, 1] 
    assert response.json()[0]['x31_germany'] == 1
    assert response.json()[0]['x5_tuesday'] == 1
    
def test_default_valid_post_response():
  #test input with empty JSON payload
  #model will use mean imputation and default values for dummy features

    response = test_client.post(
        "/predict",
        json={"x12": "",
              "x44":"",
              "x53":"",
              "x56":"",
              "x58":"",
              "x62": "",
              "x91":"",
              "x5":"",
              "x31":"",
              "x81":""
              }
    )
    assert response.status_code == 200
    assert list(response.json()[0].keys()) == sorted(['business_outcome', 'phat', 'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', \
    'x81_October', 'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', \
    'x53', 'x81_November', 'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', \
    'x58', 'x56'])
    assert len(response.json()[0]) == 27 #27 fields expected in response; business_outcome, phat, and 25 of the model features
    assert response.json()[0]['business_outcome'] in [0, 1]
    assert response.json()[0]['phat'] == 0.5
      

def test_invalid_post_response():
  #test input with JSON array/list in one of the fields
  #expect error as x12 field must be string, float, or None

    response = test_client.post(
        "/predict",
        json={"x12": ["I am a list"],
              "x44":"0.98441208",
              "x53":"0.33612654",
              "x56":"0.137700027",
              "x58":"0.228881625",
              "x62": "0.373941237",
              "x91":"0.04191",
              "x5":"tuesday",
              "x31":"germany",
              "x81":"April"
              }
    )
    assert response.status_code != 200
    
def test_empty_json_invalid_post_response():
  #test input with empty JSON payload
    response = test_client.post(
        "/predict",
        json={}
    )
    assert response.status_code != 200
    
def test_empty_json_invalid_post_response2():
  #test input with empty JSON payload
    response = test_client.post(
        "/predict",
        json={"x1": "1"
        }
    )
    assert response.status_code != 200
    
    
def test_batch_input_valid_post_response():
  response = test_client.post(
        "/predict",
        json=[{"x12":"6882.34",
               "x44":"0.98441208",
               "x53":"0.33612654",
               "x56":"0.137700027",
               "x58":"0.228881625",
               "x62":"0.373941237",
               "x91":"0.04191",
               "x5":"tuesday",
               "x31":"germany",
               "x81":"April"
               }, 
              {"x12":"6882.34",
                "x44":"0.98441208",
                "x53":"0.33612654",
                "x56":"0.137700027",
                "x58":"0.228881625",
                "x62":"0.373941237",
                "x91":"0.04191",
                "x5":"tuesday",
                "x31":"germany",
                "x81":"May"
                }, 
                {"x12":"5002.34",
                "x44":"0.98441208",
                "x53":"0.33612654",
                "x56":"0.137700027",
                "x58":"0.228881625",
                "x62":"0.373941237",
                "x91":"0.04191",
                "x5":"tuesday",
                "x31":"asia",
                "x81":"October"
                }, 
                {"x12":"4882.34",
                 "x44":"0.98441208",
                 "x53":"0.33612654",
                 "x56":"0.137700027",
                 "x58":"0.228881625",
                 "x62":"0.373941237",
                 "x91":"0.06291",
                 "x5":"tuesday",
                 "x31":"germany",
                 "x81":"September"
                 }]
    )
  assert response.status_code == 200
  for i in response.json():
    assert i['business_outcome'] in [0, 1] 
    assert list(i.keys()) == sorted(['business_outcome', 'phat', 'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', \
    'x81_October', 'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', \
    'x53', 'x81_November', 'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', \
    'x58', 'x56'])
    assert len(i) == 27 #27 fields expected in response; business_outcome, phat, and 25 of the model features
  assert len(response.json()) == 4 #4 JSON records w/predictions expected in response for batch input of 4 records
