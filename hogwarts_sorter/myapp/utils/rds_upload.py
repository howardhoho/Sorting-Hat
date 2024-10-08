import boto3
import json
from botocore.exceptions import ClientError
import mysql.connector

def get_secret():
    secret_name = "rds-cred" 
    region_name = "us-east-2" 

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        # Retrieve the secret from Secrets Manager
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    # Parse the secret string into a Python dictionary
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret

def insert_into_db(filename, s3url, prediction):
    
    secret = get_secret()
    
    DB_HOST = secret['host']
    DB_USER = secret['username']
    DB_PASSWORD = secret['password']
    DB_PORT = secret['port']
    DB_NAME = "sorting_hat"  

    try:
        # Connect to the MySQL database using the retrieved credentials
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()
        
        print("successfully connected to MySQL")

        # Perform the database insert operation
        insert_query = """
        INSERT INTO image_predictions (filename, s3url, prediction) 
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (filename, s3url, prediction))
        
        print("successfully insert the value")

        # Commit the transaction
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as err:
        raise RuntimeError(f"Error: {err}")


