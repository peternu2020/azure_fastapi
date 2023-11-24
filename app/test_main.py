from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)

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
    assert response.json() == [{'business_outcome': 0, 
                                'phat': 0.08938371087386138, 
                                'x12': 1, 
                                'x31_asia': 0, 
                                'x31_germany': 1, 
                                'x31_japan': 0, 
                                'x44': 1, 
                                'x53': 0, 
                                'x56': 0, 
                                'x58': 0, 
                                'x5_monday': 0, 
                                'x5_saturday': 0, 
                                'x5_sunday': 0, 
                                'x5_tuesday': 1, 
                                'x62': 0, 
                                'x81_August': 0, 
                                'x81_December': 0, 
                                'x81_February': 0, 
                                'x81_January': 0, 
                                'x81_July': 0, 
                                'x81_June': 0, 
                                'x81_March': 0, 
                                'x81_May': 0, 
                                'x81_November': 0, 
                                'x81_October': 0, 
                                'x81_September': 0, 
                                'x91': 0
                                }]

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
    assert response.json() == [{'business_outcome': 0, 
                                'phat': 0.08938371087386138, 
                                'x12': 1, 
                                'x31_asia': 0, 
                                'x31_germany': 1, 
                                'x31_japan': 0, 
                                'x44': 1, 
                                'x53': 0, 
                                'x56': 0, 
                                'x58': 0, 
                                'x5_monday': 0, 
                                'x5_saturday': 0, 
                                'x5_sunday': 0, 
                                'x5_tuesday': 1, 
                                'x62': 0, 
                                'x81_August': 0, 
                                'x81_December': 0, 
                                'x81_February': 0, 
                                'x81_January': 0, 
                                'x81_July': 0, 
                                'x81_June': 0, 
                                'x81_March': 0, 
                                'x81_May': 0, 
                                'x81_November': 0, 
                                'x81_October': 0, 
                                'x81_September': 0, 
                                'x91': 0
                                }]
                                
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
    for i in response.json():
      assert i['business_outcome'] in [0, 1] 
      
      
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
    for i in response.json():
      assert i['business_outcome'] in [0, 1] 
      

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
    
def test_empty_json_valid_post_response():
  #test input with empty JSON payload
  #model will use mean imputation and default values for dummy features
    response = test_client.post(
        "/predict",
        json={
              }
    )
    assert response.status_code == 200
    assert response.json()[0]['phat'] == 0.5
    assert 1 == 2
    
    
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
  assert len(response.json()) == 4
