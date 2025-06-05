using Jive
@useinside Main module test_facedetection_haar

using Test
using FaceDetection
using .FaceDetection: FaceDetection as FD
using ColorTypes: Gray
using ImageCore: clamp01nan
using FileIO: File, @format_str, save

const img_path = normpath(@__DIR__, "assets/a1.png")
const img = FD.load_image(img_path, scale = true, scale_to = (24, 24))
@test img isa IntegralArray{Gray{Float32}, 2, Matrix{Gray{Float32}}}

const output_path = normpath(@__DIR__, "assets/output.png")
const output_img = Gray.(map(clamp01nan, img))
# save(File{format"PNG"}(output_path), output_img)


#=
  load_image(image_path::String) -> Array{Float64, N}
  Loads an image as gray_scale
    •  image_path::String: Path to an image
  Returns  IntegralArray{Float64, N}: An array of floating point values representing the image

  iA = IntegralArray([buffer,] A)
  Construct the integral array of the input array A. If buffer of the same shape as A is provided, then this is a non-allocating version.
  The integral array is calculated by assigning to each cell the sum of all cells above it and to its left, i.e. the rectangle from origin
  point to the pixel position. For example, in 1-D case, iA[i] = sum(A[1:i]) if the vector A's origin is 1.

  https://en.wikipedia.org/wiki/Haar-like_feature
  HaarLikeObject(
      feature_type::Tuple{Integer, Integer},
      position::Tuple{Integer, Integer},
      width::Integer,
      height::Integer,
      threshold::Integer,
      polarity::Integer
  ) -> HaarLikeObject
      Struct representing a Haar-like feature.

  FEATURE_TYPES
(two_vertical = (1, 2), two_horizontal = (2, 1), three_horizontal = (3, 1), three_vertical = (1, 3), four = (2, 2))

  get_faceness(feature::HaarLikeObject{I, F}, int_img::IntegralArray{T, N}) -> Number
  Get facelikeness for a given feature.
    •  feature::HaarLikeObject: given Haar-like feature (parameterised replacement of Python's self)
    •  int_img::IntegralArray: Integral image array
    Returns  score::Number: Score for given feature

  sum_region(
  	integral_image_arr::AbstractArray,
  	top_left::Tuple{Int,Int},
  	bottom_right::Tuple{Int,Int}
  ) -> Number
    •  iA::IntegralArray{T, N}: The intermediate Integral Image
    •  top_left::NTuple{N, Int}: coordinates of the rectangle's top left corner
    •  bottom_right::NTuple{N, Int}: coordinates of the rectangle's bottom right corner
    Returns  sum::T The sum of all pixels in the given rectangle defined by the parameters top_left and bottom_right

  get_vote(feature::HaarLikeObject, int_img::IntegralArray) -> Integer
  Get vote of this feature for given integral image.
    •  feature::HaarLikeObject: given Haar-like feature
    •  int_img::IntegralArray: Integral image array
    Returns  vote::Integer: 1 ⟺ this feature votes positively -1 otherwise

  ensemble_vote_all(images::Vector{String}, classifiers::Vector{HaarLikeObject}) -> Vector{Int8}
  ensemble_vote_all(image_path::String, classifiers::Vector{HaarLikeObject})     -> Vector{Int8}
  Given a path to images, loads images then classifies votes using given classifiers.
  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1);
        else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.
    •  images::Vector{String}: list of paths to images; OR image_path::String: Path to images dir
    •  classifiers::Vector{HaarLikeObject}: List of classifiers
    Returns  votes::Vector{Int8}: A list of assigned votes (see ensemble_vote).

  FD.create_features(
      img_height::Int, img_width::Int,
      min_feature_width::Int,
      max_feature_width::Int,
      min_feature_height::Int,
      max_feature_height::Int
  ) -> Array{HaarLikeObject, 1}
  Iteratively creates the Haar-like feautures
    •  img_height::Integer: The height of the image
    •  img_width::Integer: The width of the image
    •  min_feature_width::Integer: The minimum width of the feature (used for computation efficiency purposes)
    •  max_feature_width::Integer: The maximum width of the feature
    •  min_feature_height::Integer: The minimum height of the feature
    •  max_feature_height::Integer: The maximum height of the feature
    Returns  features::AbstractArray: an array of Haar-like features found for an image


  determine_feature_size(
      pictures::Vector{String}
  ) -> Tuple{Integer, Integer, Integer, Integer, Tuple{Integer, Integer}}

  determine_feature_size(
      pos_training_path::String,
      neg_training_path::String
  ) -> Tuple{Integer, Integer, Integer, Integer, Tuple{Integer, Integer}}

  Takes images and finds the best feature size for the image size.
    •  pictures::Vector{String}: a list of paths to the images
  OR
    •  pos_training_path::String: the path to the positive training
       images
    •  neg_training_path::String: the path to the negative training
       images
  Returns
    •  max_feature_width::Integer: the maximum width of the feature
    •  max_feature_height::Integer: the maximum height of the feature
    •  min_feature_height::Integer: the minimum height of the feature
    •  min_feature_width::Integer: the minimum width of the feature
    •  min_size_img::Tuple{Integer, Integer}: the minimum-sized image in
       the image directories


get_feature_votes(positive_files::Vector{String}, negative_files::Vector{String}; ...)
        num_classifiers,
        min_feature_height,
        max_feature_height,
        min_feature_width,
        max_feature_width;
        scale,
        scale_to,

reconstruct
reconstruct(classifiers::Array{HaarLikeObject{I, F}, 1}, img_size::Tuple{Int64, Int64}) where {I, F}
=#

end # module test_facedetection_haar
