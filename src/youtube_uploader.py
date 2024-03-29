import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "/home/ubuntu/tracker/tmp/client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    flow.run_local_server(port=8081, open_browser=True)
    session = flow.authorized_session()
    # token = flow.fetch_token()
    
    
    return build(API_SERVICE_NAME, API_VERSION, credentials = session.credentials)

def create_playlist(youtube, title, description):
    playlists_insert_response = youtube.playlists().insert(
        part="snippet,status",
        body=dict(
            snippet=dict(
                title=title,
                description=description
            ),
            status=dict(
                privacyStatus="private"
            )
        )
    ).execute()

    return playlists_insert_response["id"]

def upload_video(youtube, video_file, title, description, category, tags, playlist_id):
    body=dict(
        snippet=dict(
            title=title,
            description=description,
            tags=tags,
            categoryId=category
        ),
        status=dict(
            privacyStatus='private'
        )
    )

    media = MediaFileUpload(video_file, chunksize=-1, resumable=True)

    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media
    )

    response = request.execute()

    youtube.playlistItems().insert(
        part="snippet",
        body={
            'snippet': {
                'playlistId': playlist_id, 
                'resourceId': {
                    'kind': 'youtube#video',
                    'videoId': response['id']
                }
            }
        }
    ).execute()

def main():
    youtube = get_authenticated_service()
    playlist_id = create_playlist(youtube, "Yeladot_HrzlVsNatanya", "")
    print(f"List created playlist_id - {playlist_id}")

    directory = "output/nat_final"
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):  # assuming you're uploading .mp4 files
            file_path = os.path.join(directory, filename)
            print(f"Uploading {file_path}")
            upload_video(youtube, file_path, filename, filename, "", "", playlist_id)
    # upload_video(youtube, "YOUR_VIDEO_FILE_PATH", "YOUR_VIDEO_TITLE", "YOUR_VIDEO_DESCRIPTION", "YOUR_VIDEO_CATEGORY", "YOUR_VIDEO_TAGS", playlist_id)

if __name__ == "__main__":
    main()