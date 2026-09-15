import cv2


def sift_feature_matching(image1, image2):
    """
    Detect and match SIFT features between two images.
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(
        gray1,
        None
    )

    keypoints2, descriptors2 = sift.detectAndCompute(
        gray2,
        None
    )

    if descriptors1 is None or descriptors2 is None:
        return None, 0

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    result = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return result, len(good_matches)