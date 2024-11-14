synthetic_data_json_llama1B = """
 {
 "results": [
      {
         "age": 36.437,
         "sex": "male",
         "bmi": 31.281,
         "children": 3,
         "smoker": "yes",
         "region": "southeast",
         "charges": 35715.219
      },
      {
         "age": 45.141,
         "sex": "female",
         "bmi": 27.943,
         "children": 2,
         "smoker": "no",
         "region": "northwest",
         "charges": 34601.439
      },
      {
         "age": 29.354,
         "sex": "male",
         "bmi": 22.357,
         "children": 2,
         "smoker": "no",
         "region": "southeast",
         "charges": 19818.819
      },
      {
         "age": 47.149,
         "sex": "female",
         "bmi": 26.811,
         "children": 3,
         "smoker": "yes",
         "region": "northwest",
         "charges": 34871.399
      },
      {
         "age": 56.119,
         "sex": "male",
         "bmi": 32.513,
         "children": 1,
         "smoker": "yes",
         "region": "northeast",
         "charges": 31432.939
      },
      {
         "age": 34.495,
         "sex": "female",
         "bmi": 28.301,
         "children": 2,
         "smoker": "no",
         "region": "southwest",
         "charges": 25360.469
      },
      {
         "age": 51.351,
         "sex": "male",
         "bmi": 31.411,
         "children": 3,
         "smoker": "no",
         "region": "northeast",
         "charges": 26733.489
      },
      {
         "age": 46.187,
         "sex": "female",
         "bmi": 28.299,
         "children": 1,
         "smoker": "no",
         "region": "southwest",
         "charges": 25730.359
      },
      {
         "age": 52.939,
         "sex": "male",
         "bmi": 34.611,
         "children": 2,
         "smoker": "no",
         "region": "northeast",
         "charges": 26533.459
      },
      {
         "age": 41.123,
         "sex": "female",
         "bmi": 27.627,
         "children": 3,
         "smoker": "yes",
         "region": "southwest",
         "charges": 24343.919
      },
      {
         "age": 44.371,
         "sex": "male",
         "bmi": 33.131,
         "children": 1,
         "smoker": "no",
         "region": "northeast",
         "charges": 20533.209
      },
      {
         "age": 22.221,
         "sex": "female",
         "bmi": 26.329,
         "children": 2,
         "smoker": "no",
         "region": "northwest",
         "charges": 14539.489
      },
      {
         "age": 37.229,
         "sex": "male",
         "bmi": 32.839,
         "children": 2,
         "smoker": "yes",
         "region": "southwest",
         "charges": 23739.479
      },
      {
         "age": 37.039,
         "sex": "female",
         "bmi": 26.249,
         "children": 2,
         "smoker": "no",
         "region": "northeast",
         "charges": 19745.799
      },
      {
         "age": 24.941,
         "sex": "male",
         "bmi": 30.691,
         "children": 2,
         "smoker": "no",
         "region": "northwest",
         "charges": 15981.049
      },
      {
         "age": 70.471,
         "sex": "female",
         "bmi": 40.131,
         "children": 3,
         "smoker": "yes",
         "region": "southeast",
         "charges": 32044.219
      },
      {
         "age": 51.931,
         "sex": "male",
         "bmi": 33.329,
         "children": 2,
         "smoker": "no",
         "region": "southwest",
         "charges": 29121.879
      },
      {
         "age": 47.431,
         "sex": "female",
         "bmi": 25.359,
         "children": 3,
         "smoker": "yes",
         "region": "northeast",
         "charges": 23546.749
      },
      {
         "age": 47.659,
         "sex": "male",
         "bmi": 39.209,
         "children": 1,
         "smoker": "no",
         "region": "southwest",
         "charges": 23975.401
      },
      {
         "age": 33.951,
         "sex": "female",
         "bmi": 29.831,
         "children": 2,
         "smoker": "yes",
         "region": "northwest",
         "charges": 18180.149
      },
      {
         "age": 44.591,
         "sex": "male",
         "bmi": 30.189,
         "children": 1,
         "smoker": "no",
         "region": "northeast",
         "charges": 20541.499
      },
      {
         "age": 50.931,
         "sex": "female",
         "bmi": 33.241,
         "children": 2,
         "smoker": "no",
         "region": "southwest",
         "charges": 24993.329
      }
    ]
}
"""

# Option 1: Paste CSV-formatted string outputted by LLM in {name}_model_deployment.py and have script automatically write to csv file for synth data
# Option 2: Skip this step and manually copy and paste the outputted CSV-formatted string of synthetic data in the csv file directly
synthetic_csv_formatted_data = """
"age","sex","bmi","children","smoker","region","charges"
22,"male",32.91,0,"no","southeast",3170.345
41,"female",24.49,2,"no","northeast",7962.302
30,"male",40.61,1,"yes","northwest",49459.188
19,"female",36.19,0,"yes","southwest",12658.201
38,"male",33.45,3,"no","southeast",10882.159
50,"female",29.76,1,"no","northeast",6382.832
25,"male",35.67,0,"yes","southwest",36419.463
63,"male",24.33,0,"no","northwest",27602.395
56,"female",33.37,2,"yes","southeast",44578.981
42,"male",22.09,1,"no","northeast",7050.592
19,"female",39.21,0,"no","northeast",2665.399
39,"male",41.41,1,"yes","southwest",45781.369
23,"female",24.04,0,"yes","northwest",17278.417
50,"female",31.55,0,"no","southeast",5134.339
27,"male",37.61,3,"no","northeast",6603.196
35,"male",28.21,0,"yes","northwest",34587.295
46,"male",24.05,2,"no","southeast",7802.551
44,"male",39.92,0,"no","northeast",10454.599
62,"female",34.25,0,"no","southwest",14563.196
21,"male",30.84,0,"yes","southeast",23441.689
58,"female",31.94,0,"no","northeast",9903.516
41,"male",26.73,2,"no","northwest",6905.669
32,"female",36.79,1,"yes","southeast",46921.551
26,"male",18.13,0,"no","northwest",2491.372
38,"female",33.46,0,"yes","northwest",40967.219
46,"male",31.17,1,"no","southwest",7957.398
60,"male",25.28,0,"yes","northwest",31234.697
54,"female",29.85,0,"yes","northeast",24544.469
29,"male",42.52,1,"yes","southeast",51726.411
17,"female",36.33,0,"no","southeast",2423.455
42,"male",21.38,3,"yes","southwest",34296.699
55,"female",38.99,1,"no","northeast",8553.169
36,"male",40.73,2,"no","northwest",6311.139
31,"male",34.68,0,"no","southeast",4306.415
43,"female",30.53,2,"yes","southwest",39128.661
21,"male",25.84,0,"yes","southeast",21058.352
26,"female",24.58,0,"yes","northwest",19560.549
41,"female",36.48,1,"yes","southeast",31618.911
37,"male",28.29,3,"no","northeast",5953.855
59,"male",31.08,0,"yes","southwest",28917.955
24,"male",35.85,1,"yes","northwest",41775.497
34,"female",38.41,0,"no","southeast",5000.449
48,"female",24.91,0,"no","northeast",8216.339
22,"male",38.55,0,"yes","southwest",36816.191
64,"female",29.39,0,"no","northwest",10799.285
58,"male",35.49,0,"yes","southeast",36234.882
42,"female",28.19,2,"no","southwest",7666.381
35,"male",37.26,1,"no","northwest",6903.352
29,"female",23.83,0,"yes","northeast",25021.299
54,"female",28.33,0,"yes","northwest",32052.403
55,"male",31.61,2,"yes","southeast",38586.599
44,"male",44.52,0,"no","southwest",10583.145
61,"male",35.44,0,"yes","southeast",45895.462
50,"male",36.35,1,"yes","northwest",55268.523
25,"female",38.97,0,"yes","northwest",30184.941
33,"male",33.41,2,"yes","southeast",40714.849
56,"female",25.72,2,"no","northwest",8292.482
27,"male",30.62,0,"yes","northwest",30291.685
60,"female",39.45,0,"yes","northwest",46280.145
34,"male",36.24,2,"yes","southwest",46763.345
58,"male",26.59,0,"no","northeast",10737.195
63,"female",29.63,1,"yes","northwest",28890.399
50,"female",23.29,0,"no","southwest",5751.333
27,"male",42.27,1,"no","northwest",7285.253
24,"male",29.77,0,"yes","southeast",25649.542
32,"female",29.38,0,"no","northeast",4154.299
58,"male",37.99,1,"no","northwest",8796.304
44,"female",34.18,0,"no","northeast",7143.188
38,"male",29.32,2,"yes","northwest",40122.331
52,"female",30.08,0,"no","northwest",10758.299
34,"male",40.79,0,"no","northeast",11241.549
47,"male",20.65,1,"no","southwest",5972.188
56,"male",39.61,0,"yes","northeast",49838.289
38,"female",27.34,2,"no","northwest",7335.349
55,"male",31.82,1,"yes","northwest",32338.499
28,"female",28.08,0,"yes","northeast",17731.395
25,"female",38.51,0,"no","northeast",5083.351
24,"male",22.86,1,"no","southeast",4927.165
29,"female",25.41,0,"yes","southwest",26563.517
62,"male",29.34,0,"no","northwest",14230.245
48,"female",26.04,0,"no","southwest",10576.321
62,"male",34.65,0,"yes","northwest",37165.587
46,"female",28.11,0,"yes","southwest",38628.449
40,"male",42.36,0,"no","northwest",10255.199
33,"male",38.29,1,"yes","southwest",47561.181
31,"female",26.55,0,"no","northeast",4324.362
26,"male",33.67,2,"yes","northwest",40523.403
21,"female",40.12,0,"yes","northwest",29638.313
43,"male",39.28,2,"yes","northeast",44972.411
56,"female",27.55,1,"yes","southwest",40745.345
51,"male",30.49,0,"no","northwest",10643.259
25,"male",28.2,1,"no","northwest",4655.195
"""